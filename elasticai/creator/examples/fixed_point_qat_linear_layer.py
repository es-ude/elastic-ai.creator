from copy import deepcopy
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset

from elasticai.creator.vhdl.evaluators.inference_evaluator import (
    QuantizedInferenceEvaluator,
)
from elasticai.creator.vhdl.evaluators.metric_evaluator import MetricEvaluator
from elasticai.creator.vhdl.number_representations import FixedPoint, FixedPointFactory
from elasticai.creator.vhdl.quantized_modules import (
    FixedPointHardSigmoid,
    FixedPointLinear,
    FixedPointReLU,
)
from elasticai.creator.vhdl.quantized_modules.autograd_functions import (
    FixedPointDequantFunction,
    FixedPointQuantFunction,
)


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
    ds_train: Dataset,
    ds_val: Dataset,
    batch_size: int,
    learning_rate: float,
    num_epochs: int,
    fp_factory: FixedPointFactory,
) -> tuple[list[float], list[float], list[float]]:
    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    dl_val = DataLoader(ds_val, batch_size=batch_size)

    loss_fn = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    accuracy_metric = partial(binary_accuracy, threshold=0.5)

    quantized_inference_evaluator = QuantizedInferenceEvaluator(
        module=model,
        data=ds_val,
        input_quant=lambda x: FixedPointQuantFunction.apply(x, fp_factory),
        output_dequant=lambda x: FixedPointDequantFunction.apply(x, fp_factory),
    )
    quantized_loss_evaluator = MetricEvaluator(
        inference_evaluator=quantized_inference_evaluator, metric=loss_fn
    )
    quantized_accuracy_evaluator = MetricEvaluator(
        inference_evaluator=quantized_inference_evaluator, metric=accuracy_metric
    )

    train_loss_per_epoch = []
    val_loss_per_epoch = []
    val_accuracy_per_epoch = []

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

        train_loss_per_epoch.append(running_train_loss / len(dl_train))

        model.train(False)

        running_val_loss = 0.0
        running_accuracy = 0.0

        for inputs, labels in dl_val:
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)

            running_val_loss += loss.item()
            running_accuracy += accuracy_metric(outputs, labels)

        val_loss_per_epoch.append(running_val_loss / len(dl_val))
        val_accuracy_per_epoch.append(running_accuracy / len(dl_val))

        print(
            f"[epoch {epoch + 1}]",
            f"loss: {train_loss_per_epoch[epoch]:.04};",
            f"val_loss: {val_loss_per_epoch[epoch]:.04};",
            f"val_acc: {val_accuracy_per_epoch[epoch]:.04}",
        )

    simulated_quant_val_loss = quantized_loss_evaluator.run()
    simulated_quant_val_acc = quantized_accuracy_evaluator.run()

    print(
        "[training summary]",
        f"\tloss: {train_loss_per_epoch[-1]:.04};",
        f"\tval_loss: {val_loss_per_epoch[-1]:.04};",
        f"\tval_acc: {val_accuracy_per_epoch[-1]:.04}",
        f"\tsimulated_quant_val_loss: {simulated_quant_val_loss:.04};",
        f"\tsimulated_quant_val_acc: {simulated_quant_val_acc:.04}",
        sep="\n",
    )

    return train_loss_per_epoch, val_loss_per_epoch, val_accuracy_per_epoch


def plot_params(init_model: torch.nn.Module, final_model: torch.nn.Module) -> None:
    def get_params(model: torch.nn.Module) -> np.ndarray:
        return np.concatenate(
            [param.detach().numpy().flatten() for param in model.parameters()]
        )

    init_params = get_params(init_model)
    final_params = get_params(final_model)
    param_indices = np.arange(len(init_params))

    bar_width = 0.4
    plt.bar(param_indices - 0.2, init_params, bar_width, label="initial params")
    plt.bar(param_indices + 0.2, final_params, bar_width, label="final params")
    plt.legend()
    plt.show()


def plot_loss_curve(
    train_losses: list[float], val_losses: list[float], val_accuracy: list[float]
) -> None:
    plt.plot(train_losses, label="train loss")
    plt.plot(val_losses, label="validation loss")
    plt.plot(val_accuracy, label="validation accuracy")
    plt.legend()
    plt.show()


class FixedPointModel(torch.nn.Module):
    def __init__(self, fixed_point_factory: FixedPointFactory) -> None:
        super().__init__()
        self._linear1 = FixedPointLinear(
            in_features=3, out_features=2, fixed_point_factory=fixed_point_factory
        )
        self._linear2 = FixedPointLinear(
            in_features=2, out_features=1, fixed_point_factory=fixed_point_factory
        )
        self._relu = FixedPointReLU(fixed_point_factory=fixed_point_factory)
        self._sigmoid = FixedPointHardSigmoid(fixed_point_factory=fixed_point_factory)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._linear1(x)
        x = self._relu(x)
        x = self._linear2(x)
        x = self._sigmoid(x)
        return x

    def quantized_forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._linear1.quantized_forward(x)
        x = self._relu.quantized_forward(x)
        x = self._linear2.quantized_forward(x)
        x = self._sigmoid.quantized_forward(x)
        return x


def main() -> None:
    x, y = get_dataset()
    x_train, y_train = augment_data(x, y, expand_factor=240, noise_var=0.05, seed=24)
    x_test, y_test = augment_data(x, y, expand_factor=160, noise_var=0.05, seed=24)
    ds_train = TensorDataset(x_train, y_train)
    ds_test = TensorDataset(x_test, y_test)

    fixed_point_factory = FixedPoint.get_factory(total_bits=8, frac_bits=4)
    final_model = FixedPointModel(fixed_point_factory)
    init_model = deepcopy(final_model)

    history = train(
        model=final_model,
        ds_train=ds_train,
        ds_val=ds_test,
        batch_size=16,
        learning_rate=1e-3,
        num_epochs=200,
        fp_factory=fixed_point_factory,
    )

    plot_loss_curve(*history)
    plot_params(init_model, final_model)


if __name__ == "__main__":
    main()
