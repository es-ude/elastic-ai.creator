import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

from elasticai.creator.base_modules.arithmetics.fixed_point_arithmetics import (
    FixedPointArithmetics,
)
from elasticai.creator.base_modules.two_complement_fixed_point_config import (
    FixedPointConfig,
)
from elasticai.creator.nn.vhdl import FPBatchNormedLinear, FPLinear, Sequential


def get_dataset(samples_per_cluster: int = 500) -> TensorDataset:
    xs_c0 = torch.empty(samples_per_cluster, 1).normal_(mean=25, std=2)
    ys_c0 = torch.empty(samples_per_cluster, 1).normal_(mean=3, std=0.5)
    samples_c0 = torch.cat([xs_c0, ys_c0], dim=1)
    labels_c0 = torch.ones(samples_per_cluster) * 0

    xs_c1 = torch.empty(samples_per_cluster, 1).normal_(mean=1, std=6)
    ys_c1 = torch.empty(samples_per_cluster, 1).normal_(mean=-0.5, std=0.2)
    samples_c1 = torch.cat([xs_c1, ys_c1], dim=1)
    labels_c1 = torch.ones(samples_per_cluster) * 1

    xs_c2 = torch.empty(samples_per_cluster, 1).normal_(mean=-0.5, std=8)
    ys_c2 = torch.empty(samples_per_cluster, 1).normal_(mean=2, std=0.6)
    samples_c2 = torch.cat([xs_c2, ys_c2], dim=1)
    labels_c2 = torch.ones(samples_per_cluster) * 2

    samples = torch.cat([samples_c0, samples_c1, samples_c2])
    labels = torch.cat([labels_c0, labels_c1, labels_c2]).to(torch.int64)

    return TensorDataset(samples, labels)


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


class Model(torch.nn.Module):
    def __init__(self, total_bits=16, frac_bits=8) -> None:
        super().__init__()
        self.arithmetics = FixedPointArithmetics(
            FixedPointConfig(total_bits, frac_bits)
        )
        self.bn_lin = FPBatchNormedLinear(
            in_features=2,
            out_features=10,
            bias=True,
            total_bits=total_bits,
            frac_bits=frac_bits,
        )
        self.lin = FPLinear(
            in_features=10,
            out_features=3,
            bias=True,
            total_bits=total_bits,
            frac_bits=frac_bits,
        )
        self.model = Sequential(
            self.bn_lin, torch.nn.ReLU(), self.lin, torch.nn.Softmax(dim=1)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = self.arithmetics.quantize(inputs)
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
        epochs=50,
        loss_fn=torch.nn.CrossEntropyLoss(),
    )

    samples, _ = ds[:]
    preds = model(samples).argmax(dim=1)

    xs, ys = np.split(samples.numpy(), 2, axis=1)
    plt.scatter(xs.flatten(), ys.flatten(), c=preds.detach().numpy())
    plt.show()


if __name__ == "__main__":
    main()
