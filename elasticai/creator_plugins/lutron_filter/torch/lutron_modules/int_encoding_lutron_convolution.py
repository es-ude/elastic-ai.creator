from typing import Any, Callable

import torch
from torch import Tensor
from torch.nn import BatchNorm1d, Conv1d, Module

from elasticai.creator.ir.helpers import FilterParameters
from elasticai.creator_plugins.lutron_filter.nn.lutron.truth_table_generation import (
    generate_input_tensor,
)

from .lutron_filter import LutronFilter


class IntEncodingLutronConv1d(Module, LutronFilter):
    def __init__(self, wrapped: Conv1d, bits: int, binarize: Module):
        super().__init__()
        self.bits = bits
        self.wrapped = wrapped
        self.bin = binarize
        self.batchnorm = BatchNorm1d(wrapped.out_channels)
        self.lutron_parameters = FilterParameters(
            kernel_size=wrapped.kernel_size[0],
            in_channels=bits,
            out_channels=wrapped.out_channels,
            groups=wrapped.groups,
            stride=wrapped.stride[0],
        )
        self._creator_meta = {
            "type": "lutron_filter",
            "filter_parameters": self.lutron_parameters,
        }
        self._forward = self._training_forward

    @property
    def creator_meta(self) -> dict[str, Any]:
        return self._creator_meta

    @property
    def filter_parameters(self) -> FilterParameters:
        return self._creator_meta["filter_parameters"]

    def infer_shape(
        self,
        input_shape: tuple[int, int, int],
        output_shape: tuple[int, int, int],
    ) -> None:
        self.lutron_parameters.input_size = input_shape[2]
        self.lutron_parameters.output_size = output_shape[2]
        self.lutron_parameters.in_channels = input_shape[1]
        self.lutron_parameters.out_channels = output_shape[1]

    def forward(self, x: Tensor) -> Tensor:
        return self._forward(x)

    def generate_io_tensors(self) -> tuple[Tensor, Tensor]:
        inputs = generate_input_tensor(self.bits, kernel_size=1, groups=1)
        outputs = self(inputs)
        return inputs, outputs

    def _translation_forward(self, x: Tensor) -> Tensor:
        return self._training_forward(convert_lutron_bits_to_ints_as_two_complements(x))

    def _training_forward(self, input: Tensor) -> Tensor:
        x = self.bin(self.batchnorm(self.wrapped(input)))
        return x

    def prepare_for_translation(self):
        self._forward = self._translation_forward

    def train(self, mode: bool = True) -> "IntEncodingLutronConv1d":
        super().train(mode)
        if mode:
            self._forward = self._translation_forward
        return self


def _create_weight_for_conversion_as_twos_complement(bits: int) -> Tensor:
    return torch.tensor(
        [[[-(2 ** (bits - 1))]] + [[2**i] for i in reversed(range(0, bits - 1))]],
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
    c = Conv1d(
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
