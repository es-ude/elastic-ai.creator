from typing import Callable, Protocol, runtime_checkable

import torch
from torch import Tensor
from torch.nn import Conv1d as _Conv1d
from torch.nn import Module


def _create_weight_for_conversion_as_twos_complement(bits: int) -> Tensor:
    return torch.tensor(
        [[-(2 ** (bits - 1))]] + [[2**i] for i in reversed(range(0, bits - 1))],
        dtype=torch.float32,
    )


def _create_weight_for_conversion_as_positives(bits: int) -> Tensor:
    return torch.tensor(
        [[[2**i] for i in reversed(range(0, bits))]],
        dtype=torch.float32,
    )


def convert_lutron_bits_to_ints(
    input: Tensor, in_channels: int, weight_data: Callable[[int], Tensor]
) -> Tensor:
    if len(input.shape) == 3:
        batch_size, bits_times_in_channels, length = input.shape
    elif len(input.shape) == 2:
        bits_times_in_channels, length = input.shape
    else:
        raise ValueError("Expected input to have 2 or 3 dimensions")
    bits = bits_times_in_channels // in_channels
    c = _Conv1d(
        in_channels=bits_times_in_channels,
        out_channels=in_channels,
        kernel_size=1,
        bias=False,
        groups=in_channels,
    )
    c.weight.data = weight_data(bits).repeat(in_channels, 1, 1)
    c.to(device=input.device)
    input = (input + 1) / 2.0
    return c(input)


def convert_lutron_bits_to_positive_ints(
    input: Tensor, bits: int, in_channels: int = 1
) -> Tensor:
    return convert_lutron_bits_to_ints(
        input, in_channels, _create_weight_for_conversion_as_positives
    )


def convert_lutron_bits_to_ints_as_two_complements(
    input: Tensor, in_channels: int = 1
) -> Tensor:
    return convert_lutron_bits_to_ints(
        input, in_channels, _create_weight_for_conversion_as_twos_complement
    )


@runtime_checkable
class Conv1d(Protocol):
    kernel_size: tuple[int, ...]
    in_channels: int
    out_channels: int
    groups: int
    stride: tuple[int, ...]

    def __call__(self, x: torch.Tensor) -> torch.Tensor: ...


class LutronBitsToTwoComplementsWrapper(Module):
    def __init__(self, num_bits: int, wrapped: Conv1d):
        super().__init__()
        self._num_bits = num_bits
        self._wrapped = wrapped
        in_channels = num_bits * wrapped.in_channels
        out_channels = wrapped.in_channels
        groups = wrapped.groups * wrapped.in_channels
        self._preceding = _Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            groups=groups,
            kernel_size=1,
            stride=1,
            bias=False,
        )
        weight_data = _create_weight_for_conversion_as_twos_complement(num_bits)
        self._preceding.weight.data = weight_data.repeat(
            out_channels, wrapped.groups, 1
        )

    @property
    def in_channels(self) -> int:
        return self._preceding.in_channels

    @property
    def out_channels(self) -> int:
        return self._wrapped.out_channels

    @property
    def groups(self) -> int:
        return self._preceding.groups

    @property
    def kernel_size(self) -> int:
        return self._wrapped.kernel_size[0]

    def forward(self, x):
        x = self._preceding((x + 1) / 2)
        return self._wrapped(x)
