import torch
from torch import tensor


def generate_input_tensor_1d(in_channels, kernel_size, groups=1):
    length = in_channels // groups * kernel_size
    elements = tuple(tensor([-1.0, 1.0]) for _ in range(length))
    return (
        torch.cartesian_prod(*elements)
        .reshape(2**length, in_channels // groups, kernel_size)
        .repeat(1, groups, 1)
    )
