import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Dataset

from elasticai.creator.layers import QLSTM, QLinear


class SinusDataset(Dataset):
    def __init__(self, sinus_range=(1, 500), num_points=5000, window_size=100):
        self.sinus_range = sinus_range
        self.num_points = num_points
        self.window_size = window_size

        x = torch.linspace(*self.sinus_range, self.num_points)
        self._full_series = (
            torch.sin(x) * (self.sinus_range[1] - x) / self.sinus_range[1]
        )
        self._data = self._rolling_samples_with_labels(
            self._full_series, self.window_size
        )
        self._data = self._reshape_samples(self._data)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx]

    @staticmethod
    def _rolling_samples_with_labels(data, window_size):
        result = []
        for i in range(window_size, len(data)):
            result.append((data[i - window_size : i], data[i]))
        return result

    @staticmethod
    def _reshape_samples(data):
        return [(sample.view(-1, 1), label) for sample, label in data]

    @property
    def full_series(self):
        return self._full_series


class QLSTMModel(torch.nn.Module):
    def __init__(self, window_size):
        super().__init__()
        # NOTE: With quantization I get really high loss values. But I don't spend lot of time in quantizing this model
        self.lstm1 = QLSTM(
            input_size=1,
            hidden_size=64,
            bias=True,  # bias=False
            state_quantizer=None,  # state_quantizer=Binarize()
            weight_quantizer=None,  # weight_quantizer=Binarize()
            input_gate_activation=None,  # input_gate_activation=Binarize()
            forget_gate_activation=None,  # forget_gate_activation=Binarize()
            cell_gate_activation=None,  # cell_gate_activation=Ternarize()
            output_gate_activation=None,  # output_gate_activation=Binarize()
            new_cell_state_activation=None,  # new_cell_state_activation=Ternarize()
            batch_first=True,
        )
        #        self.bin = Binarize()
        self.linear = QLinear(
            in_features=window_size * 64,
            out_features=1,
            bias=True,  # bias=False
            quantizer=None,  # quantizer=Binarize
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x, _ = self.lstm1(inputs)
        #        x = self.bin(x)
        x = self.linear(x.view(len(x), -1))
        return x


if __name__ == "__main__":
    ds = SinusDataset(sinus_range=(1, 500), num_points=5000, window_size=100)

    # plt.plot(ds.full_series)
    # plt.show()

    num_train_samples = int(len(ds) * 0.8)
    ds_train = ds[:num_train_samples]
    ds_test = ds[num_train_samples:]

    batch_size = 32
    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    dl_test = DataLoader(ds_test, batch_size=batch_size, shuffle=True)

    print("Number of train batches:", len(dl_train))
    print("Number of test batches:", len(dl_test))

    lstm_model = QLSTMModel(ds.window_size)

    n_epochs = 10
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(lstm_model.parameters(), lr=1e-4)

    for epoch in range(n_epochs):
        losses = []

        for X, y in dl_train:
            y_pred = lstm_model(X)
            loss = loss_fn(
                y_pred.view(
                    -1,
                ),
                y,
            )
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.item())
        train_loss = sum(losses) / len(losses)

        with torch.no_grad():
            val_loss = sum(
                loss_fn(
                    lstm_model(X).view(
                        -1,
                    ),
                    y,
                )
                for X, y in dl_test
            ) / len(dl_test)

        print(
            "Epoch: {}/{}; Loss: {}; Val_Loss: {}".format(
                epoch + 1, n_epochs, train_loss, val_loss
            )
        )

    # Plot final result for one window
    idx = 1
    ws = ds.window_size
    sample = ds_test[idx][0].tolist()
    predicted = ds_test[idx][0].tolist()
    target = [target_value.item() for _, target_value in ds_test[idx : idx + ws]]

    for i in range(ws):
        y_pred = lstm_model(torch.tensor([predicted]))
        predicted = predicted[1:] + [
            y_pred.view(
                -1,
            ).tolist()
        ]
    x_axis = torch.arange(0, ws * 2)
    plt.plot(
        x_axis[:ws],
        torch.tensor(sample).view(
            -1,
        ),
        "-b",
        label="previous window",
    )
    plt.plot(
        x_axis[ws:],
        torch.tensor(target).view(
            -1,
        ),
        "-g",
        label="target prediction",
    )
    plt.plot(
        x_axis[ws:],
        torch.tensor(predicted).view(
            -1,
        ),
        "-r",
        label="actual prediction",
    )
    plt.legend()
    plt.show()
