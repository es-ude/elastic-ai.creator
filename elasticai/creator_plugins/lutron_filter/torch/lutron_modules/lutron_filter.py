from abc import ABC
from typing import Any, Protocol, Sequence

from torch import Tensor
from torch.nn import Module

from elasticai.creator.ir.helpers import FilterParameters
from elasticai.creator_plugins.lutron_filter.nn.lutron.truth_table_generation import (
    generate_input_tensor,
)

from ..lutron_module_protocol import LutronModule


class LutronFilter(LutronModule, Protocol):
    @property
    def filter_parameters(self) -> FilterParameters: ...


class _LutronFilterBase(Module, LutronModule, ABC):
    def __init__(self, wrapped: Module, filter_parameters: FilterParameters):
        super().__init__()
        self._wrapped = wrapped
        self._creator_meta = {
            "type": "lutron_filter",
            "filter_parameters": filter_parameters,
        }

    @property
    def creator_meta(self) -> dict[str, Any]:
        return self._creator_meta

    @property
    def filter_parameters(self) -> FilterParameters:
        return self.creator_meta["filter_parameters"]

    def forward(self, x: Tensor) -> Tensor:
        return self._wrapped(x)

    def generate_io_tensors(self) -> tuple[Tensor, Tensor]:
        inputs = generate_input_tensor(
            kernel_size=self.filter_parameters.kernel_size,
            groups=self.filter_parameters.groups,
            in_channels=self.filter_parameters.in_channels,
        )
        outputs = self(inputs)
        return inputs, outputs


class LutronConv(_LutronFilterBase):
    def infer_shape(
        self, input_shape: tuple[int, int, int], output_shape: tuple[int, int, int]
    ):
        batch_size, in_channels, num_points = input_shape
        batch_size, out_channels, out_size = output_shape
        p = self.filter_parameters
        p.input_size = num_points
        p.output_size = out_size


class LutronMaxPool(_LutronFilterBase):
    def infer_shape(
        self, input_shape: tuple[int, int, int], output_shape: tuple[int, int, int]
    ):
        batch_size, out_channels, num_points = input_shape
        p = self.filter_parameters
        p.in_channels = out_channels
        p.out_channels = p.in_channels
        p.groups = p.in_channels
        p.input_size = num_points
        p.output_size = output_shape[2]


class LutronLinear(_LutronFilterBase):
    def infer_shape(
        self, input_shape: Sequence[int], output_shape: Sequence[int]
    ) -> None:
        p = self.filter_parameters
        match input_shape:
            case _, in_channels, num_points:
                p.in_channels = in_channels
                p.input_size = num_points
                p.kernel_size = num_points
            case _, in_size:
                p.input_size = in_size
                p.in_channels = 1
                p.kernel_size = in_size

        match output_shape:
            case _, out_size:
                p.output_size = 1
                p.out_channels = out_size
            case _, out_channels, out_size:
                p.out_channels = out_channels
                p.output_size = out_size
