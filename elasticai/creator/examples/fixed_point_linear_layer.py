import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset

from elasticai.creator.vhdl.custom_layers.linear import FixedPointLinear
from elasticai.creator.vhdl.number_representations import FixedPoint


def get_dataset() -> tuple[torch.Tensor, torch.Tensor]:
    """
    Retruns the inputs and outputs of the following logic function:
        O = (!I1 * I2 * I3) + (I1 * !I2 * I3) + (I1 * I2 * !I3)
    """
    x = torch.as_tensor(
        [
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 1],
            [1, 0, 0],
            [1, 0, 1],
            [1, 1, 0],
            [1, 1, 1],
        ],
        dtype=torch.float32,
    )
    y = torch.as_tensor(
        [
            [0],
            [0],
            [0],
            [1],
            [0],
            [1],
            [1],
            [0],
        ],
        dtype=torch.float32,
    )
    return x, y


def augment_data(
    samples: torch.Tensor,
    labels: torch.Tensor,
    expand_factor: int,
    noise_var: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    augmented_samples = samples.tile(expand_factor, 1)
    augmented_labels = labels.tile(expand_factor, 1)
    torch.random.manual_seed(seed)
    noise = torch.randn_like(augmented_samples) * noise_var
    augmented_samples = augmented_samples + noise
    return augmented_samples, augmented_labels


def binary_accuracy(
    predictions: torch.Tensor, labels: torch.Tensor, threshold: float
) -> float:
    binary_predictions = torch.where(predictions >= threshold, 1.0, 0.0)
    true_positives = sum(
        bool(pred == label) for pred, label in zip(binary_predictions, labels)
    )
    return true_positives / len(predictions)


def train(
    model: torch.nn.Module,
    train_ds: Dataset,
    val_ds: Dataset,
    batch_size: int,
    learning_rate: float,
    num_epochs: int,
) -> tuple[list[float], list[float], list[float]]:
    dl_train = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    dl_val = DataLoader(val_ds, batch_size=batch_size)

    loss_fn = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    losses_train = []
    losses_val = []
    accuracy = []

    for epoch in range(num_epochs):
        model.train(True)

        running_train_loss = 0.0

        for inputs, labels in dl_train:
            optimizer.zero_grad()
            outputs = model(inputs)

            loss = loss_fn(outputs, labels)
            loss.backward()

            optimizer.step()
            running_train_loss += loss.item()

        losses_train.append(running_train_loss / len(dl_train))

        model.train(False)

        running_val_loss = 0.0
        running_accuracy = 0.0

        for inputs, labels in dl_val:
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)

            running_val_loss += loss.item()
            running_accuracy += binary_accuracy(outputs, labels, 0.5)

        losses_val.append(running_val_loss / len(dl_val))
        accuracy.append(running_accuracy / len(dl_val))

        print(
            f"[epoch {epoch + 1}]",
            f"loss: {losses_train[epoch]:.04};",
            f"val_loss: {losses_val[epoch]:.04};",
            f"val_acc: {accuracy[epoch]:.04}",
        )

    return losses_train, losses_val, accuracy


class FixedPointModel(torch.nn.Module):
    def __init__(self, total_bits: int, frac_bits: int) -> None:
        super().__init__()
        factory = FixedPoint.get_factory(total_bits, frac_bits)

        self._linear1 = FixedPointLinear(
            in_features=3, out_features=2, fixed_point_factory=factory
        )

        self._linear2 = FixedPointLinear(
            in_features=2, out_features=1, fixed_point_factory=factory
        )

        self._relu = torch.nn.ReLU()
        self._sigmoid = torch.nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._linear1(x)
        x = self._relu(x)
        x = self._linear2(x)
        x = self._sigmoid(x)
        return x


def main() -> None:
    x, y = get_dataset()
    x_train, y_train = augment_data(x, y, expand_factor=240, noise_var=0.05, seed=24)
    x_test, y_test = augment_data(x, y, expand_factor=160, noise_var=0.05, seed=24)
    ds_train = TensorDataset(x_train, y_train)
    ds_test = TensorDataset(x_test, y_test)

    model = FixedPointModel(total_bits=4, frac_bits=2)

    losses_train, losses_val, accuracy_val = train(
        model=model,
        train_ds=ds_train,
        val_ds=ds_test,
        batch_size=16,
        learning_rate=1e-3,
        num_epochs=100,
    )

    plt.plot(losses_train, label="train_loss")
    plt.plot(losses_val, label="val_loss")
    plt.plot(accuracy_val, label="val_accuracy")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
