import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset

from elasticai.creator.qat.layers import QLSTM, Identity, QLinear


def _x_values(value_range: tuple[float, float], sampling_rate: int) -> torch.Tensor:
    start, end = value_range
    return torch.linspace(
        start=start, end=end, steps=int((end - start) * sampling_rate)
    )


class SineDataset(Dataset):
    def __init__(
        self, sine_range: tuple[float, float], sampling_rate: int, window_size: int
    ) -> None:
        self.sine_range = sine_range
        self.sampling_rate = sampling_rate
        self.window_size = window_size

        x = _x_values(self.sine_range, self.sampling_rate)
        self._sinus = torch.sin(x)
        self._samples, self._labels = self._rolling_samples_with_labels(
            self._sinus, self.window_size
        )

    def __len__(self) -> int:
        return len(self._labels)

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        return self._samples[idx], self._labels[idx]

    @staticmethod
    def _rolling_samples_with_labels(
        data: torch.Tensor, window_size: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        samples, labels = [], []
        for i in range(window_size, len(data)):
            samples.append(data[i - window_size : i].view(-1, 1).tolist())
            labels.append(data[i].item())
        return (
            torch.tensor(samples, dtype=data.dtype),
            torch.tensor(labels, dtype=data.dtype),
        )

    @property
    def sinus_data(self):
        return self._sinus


class QLSTMModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lstm = QLSTM(
            input_size=1,
            hidden_size=64,
            bias=True,
            batch_first=True,
            state_quantizer=Identity(),
            weight_quantizer=Identity(),
        )
        self.linear = QLinear(
            in_features=64,
            out_features=1,
            bias=True,
            quantizer=Identity(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        _, (final_hidden_states, _) = self.lstm(inputs)
        x = self.linear(final_hidden_states.squeeze(0))
        return x.view(-1)


def split_dataset(dataset: Dataset, train_fraction: float) -> tuple[Dataset, Dataset]:
    num_train_samples = int(len(dataset) * train_fraction)
    ds_train = TensorDataset(*dataset[:num_train_samples])
    ds_test = TensorDataset(*dataset[num_train_samples:])
    return ds_train, ds_test


def train(
    model: torch.nn.Module,
    train_data: Dataset,
    test_data: Dataset,
    batch_size: int,
    learning_rate: float,
    epochs: int,
) -> None:
    dl_train = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    dl_test = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        losses = []
        for samples, labels in dl_train:
            predictions = model(samples)
            loss = loss_fn(predictions, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.item())
        train_loss = sum(losses) / len(losses)

        with torch.no_grad():
            val_losses = map(lambda x: loss_fn(model(x[0]), x[1]), dl_test)
            val_loss = sum(val_losses) / len(dl_test)

        print(f"[epoch: {epoch + 1}/{epochs}] loss: {train_loss}; val_loss: {val_loss}")

    model.eval()


def plot_predicted_sine(
    model: torch.nn.Module,
    prediction_range: tuple[int, int],
    sampling_rate: int,
    window_size: int,
) -> None:
    pred_start, pred_end = prediction_range
    x_prelude = _x_values(
        (pred_start - window_size / sampling_rate, pred_start), sampling_rate
    )
    x_to_predict = _x_values(prediction_range, sampling_rate)
    y_prelude = torch.sin(x_prelude)
    y_target = torch.sin(x_to_predict)

    predictions: list[float] = y_prelude.tolist()
    for i in range(int((pred_end - pred_start) * sampling_rate)):
        inputs = torch.tensor(predictions[i:]).view(-1, 1)
        prediction = model(inputs)
        predictions.append(prediction[0].item())
    y_pred = torch.tensor(predictions[window_size:])

    plt.plot(x_prelude, y_prelude, "b-", label="prelude")
    plt.plot(x_to_predict, y_target, "g-", label="target")
    plt.plot(x_to_predict, y_pred, "r--", label="prediction")
    plt.legend()
    plt.show()


def main() -> None:
    ds = SineDataset(sine_range=(0, 200), sampling_rate=16, window_size=32)
    print("Number of samples:", len(ds))

    plt.plot(ds.sinus_data)
    plt.show()

    model = QLSTMModel()
    ds_train, ds_test = split_dataset(ds, train_fraction=0.8)

    train(
        model=model,
        train_data=ds_train,
        test_data=ds_test,
        batch_size=32,
        learning_rate=1e-3,
        epochs=10,
    )

    plot_predicted_sine(
        model=model,
        prediction_range=(400, 410),
        sampling_rate=16,
        window_size=32,
    )


if __name__ == "__main__":
    main()
