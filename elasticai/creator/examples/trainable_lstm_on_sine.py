from math import ceil, floor

import torch
from matplotlib import pyplot as plt  # type: ignore[import]
from torch.utils.data import DataLoader, Dataset, random_split

from elasticai.creator.nn.linear import FixedPointLinear
from elasticai.creator.nn.lstm import FixedPointLSTMWithHardActivations
from elasticai.creator.vhdl.number_representations import FixedPoint


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
        self._samples, self._labels = self._extract_sample_label_pairs()

    def _extract_sample_label_pairs(self) -> tuple[torch.Tensor, torch.Tensor]:
        samples, labels = [], []
        for i in range(self.window_size, len(self._sinus)):
            samples.append(self._sinus[i - self.window_size : i].view(-1, 1).tolist())
            labels.append(self._sinus[i].item())
        return (
            torch.tensor(samples, dtype=self._sinus.dtype),
            torch.tensor(labels, dtype=self._sinus.dtype),
        )

    @property
    def sinus(self):
        return self._sinus

    def __len__(self) -> int:
        return len(self._labels)

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        return self._samples[idx], self._labels[idx]


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
            loss = loss_fn(predictions, labels.view(-1, 1))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.item())
        train_loss = sum(losses) / len(losses)

        with torch.no_grad():
            val_losses = map(lambda x: loss_fn(model(x[0]), x[1].view(-1, 1)), dl_test)
            val_loss = sum(val_losses) / len(dl_test)

        print(f"[epoch: {epoch + 1}/{epochs}] loss: {train_loss}; val_loss: {val_loss}")

    model.eval()


def plot_predicted_sine(
    model: torch.nn.Module,
    prediction_range: tuple[float, float],
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


class FixedPointSineModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._hidden_size = 64
        self._fp_factory = FixedPoint.get_factory(total_bits=16, frac_bits=8)
        self.lstm = FixedPointLSTMWithHardActivations(
            input_size=1,
            hidden_size=self._hidden_size,
            bias=True,
            batch_first=True,
            fixed_point_factory=self._fp_factory,
        )
        self.linear = FixedPointLinear(
            in_features=self._hidden_size,
            out_features=1,
            bias=True,
            fixed_point_factory=self._fp_factory,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (final_hidden_state, _) = self.lstm(x)
        final_hidden_state = final_hidden_state.squeeze(0)
        return self.linear(final_hidden_state)


def main() -> None:
    dataset = SineDataset(
        sine_range=(0, 10 * torch.pi), sampling_rate=16, window_size=32
    )
    ds_train, ds_test = random_split(
        dataset, [floor(len(dataset) * 0.8), ceil(len(dataset) * 0.2)]
    )
    model = FixedPointSineModel()

    train(
        model=model,
        train_data=ds_train,
        test_data=ds_test,
        batch_size=32,
        learning_rate=1e-3,
        epochs=100,
    )

    plot_predicted_sine(
        model=model,
        prediction_range=(0, 2 * torch.pi),
        sampling_rate=16,
        window_size=32,
    )


if __name__ == "__main__":
    main()
