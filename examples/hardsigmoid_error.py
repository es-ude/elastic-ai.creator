import numpy as np
import torch
from matplotlib import pyplot as plt

from elasticai.creator.arithmetic.fxp_arithmetic import FxpParams
from elasticai.creator.nn.fixed_point import HardSigmoid


def generate_input_data(total_bitwidth: int, frac_bitwidth: int) -> torch.Tensor:
    fxp = FxpParams(total_bits=total_bitwidth, frac_bits=frac_bitwidth)
    return torch.linspace(
        start=fxp.minimum_as_rational,
        end=fxp.maximum_as_rational,
        steps=2**total_bitwidth,
    )


def plot_sigmoid(
    din: np.ndarray, y_orig: np.ndarray, y_hard: np.ndarray, y_cust: np.ndarray
) -> None:
    def mbe(x: np.ndarray, y: np.ndarray) -> float:
        return sum(x - y) / x.size

    def mae(x: np.ndarray, y: np.ndarray) -> float:
        return sum(abs(x - y)) / x.size

    plt.plot(din, y_orig, color="r", label="Sigmoid")
    plt.plot(
        din,
        y_hard,
        color="k",
        label=f"HardSigmoid (PyTorch) [MAE = {mae(y_orig, y_hard):.4f}, 1e6*MBE = {1e6 * mbe(y_orig, y_hard):.4f}]",
    )
    plt.plot(
        din,
        y_cust,
        color="g",
        label=f"HardSigmoid (Creator) [MAE = {mae(y_orig, y_cust):.4f}, 1e6*MBE = {1e6 * mbe(y_orig, y_cust):.4f}]",
    )
    plt.xlabel("Input")
    plt.ylabel("Output")
    plt.grid()
    plt.tight_layout()
    plt.legend()
    plt.show(block=True)


def extract_diff(total_bitwidth: int, frac_bitwidth: int):

    sig = torch.nn.Sigmoid()
    hsig0 = torch.nn.Hardsigmoid()
    hsig1 = HardSigmoid(total_bits=total_bitwidth, frac_bits=frac_bitwidth)
    print(hsig1.get_ranges())

    din = generate_input_data(total_bitwidth=total_bitwidth, frac_bitwidth=frac_bitwidth)
    y0 = sig(din).numpy()
    y1 = hsig0(din).numpy()
    y2 = hsig1(din).numpy()

    plot_sigmoid(din=din.numpy(), y_orig=y0, y_hard=y1, y_cust=y2)


if __name__ == "__main__":
    extract_diff(
        total_bitwidth=10,
        frac_bitwidth=7
    )
