import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

from elasticai.creator.base_modules.arithmetics.fixed_point_arithmetics import (
    FixedPointArithmetics,
)
from elasticai.creator.base_modules.two_complement_fixed_point_config import (
    FixedPointConfig,
)
from elasticai.creator.nn.vhdl import FPBatchNormedLinear, FPLinear, Sequential


def get_dataset() -> TensorDataset:
    x = torch.linspace(-2, 2, 1000).reshape(-1, 1)
    y = (-0.5 * x**3 + x + 1).reshape(-1, 1)
    return TensorDataset(x, y)


def train(
    model: torch.nn.Module,
    dl_train: DataLoader,
    dl_val: DataLoader,
    epochs: int,
    loss_fn: torch.nn.Module,
    lr: float = 1e-3,
) -> None:
    train_bs = 1 if dl_train.batch_size is None else dl_train.batch_size
    val_bs = 1 if dl_val.batch_size is None else dl_val.batch_size
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train(True)

        running_loss = 0
        for samples, labels in dl_train:
            optimizer.zero_grad()

            predictions = model(samples)
            loss = loss_fn(predictions, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / train_bs

        model.train(False)

        running_loss = 0
        for samples, labels in dl_val:
            predictions = model(samples)
            loss = loss_fn(predictions, labels)

            running_loss += loss.item()

        avg_val_loss = running_loss / val_bs

        print(
            f"[epoch {epoch+1}/{epochs}] "
            f"train_loss: {avg_train_loss:.04f} ; "
            f"val_loss: {avg_val_loss:.04f}"
        )


class Quantize(torch.nn.Module):
    def __init__(self, total_bits: int, frac_bits: int) -> None:
        super().__init__()
        self.arithmetics = FixedPointArithmetics(
            FixedPointConfig(total_bits, frac_bits)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.arithmetics.quantize(x)


class Model(torch.nn.Module):
    def __init__(self, total_bits=16, frac_bits=8) -> None:
        super().__init__()
        self.quantize = Quantize(total_bits, frac_bits)
        self.bn_lin = FPBatchNormedLinear(
            in_features=1,
            out_features=100,
            bias=True,
            total_bits=total_bits,
            frac_bits=frac_bits,
        )
        self.lin = FPLinear(
            in_features=100,
            out_features=1,
            bias=True,
            total_bits=total_bits,
            frac_bits=frac_bits,
        )
        self.model = Sequential(
            self.quantize, self.bn_lin, torch.nn.Tanh(), self.quantize, self.lin
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs)


def main() -> None:
    ds = get_dataset()

    ds_train, ds_val = random_split(ds, lengths=[0.7, 0.3])
    dl_train = DataLoader(ds_train, batch_size=32)
    dl_val = DataLoader(ds_val, batch_size=32)

    model = Model()

    train(
        model,
        dl_train=dl_train,
        dl_val=dl_val,
        epochs=100,
        loss_fn=torch.nn.MSELoss(),
    )

    samples, labels = ds[:]
    preds = model(samples)
    xs, ys = samples.flatten().numpy(), labels.flatten().numpy()
    ys_pred = preds.flatten().detach().numpy()

    plt.plot(xs, ys, "r-", label="target")
    plt.scatter(xs, ys_pred, marker=".", label="prediction")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
