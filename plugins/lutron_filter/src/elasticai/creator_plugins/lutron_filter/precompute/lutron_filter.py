import copy
from abc import ABC
from typing import Sequence

from elasticai.creator_plugins.lutron_filter.nn.binarize import Binarize
from torch import Tensor
from torch.nn import Module

from elasticai.creator_plugins.grouped_filter import FilterParameters

from .lutron_module_protocol import LutronFilter as _LutronFilter
from .truth_table_generation import generate_input_tensor_1d as generate_input_tensor


def _unwrap_1d_kernel_size(k: int | tuple[int]) -> int:
    if isinstance(k, int):
        return k
    if isinstance(k, tuple) and len(k) == 1:
        return k[0]
    raise ValueError("expecting 1d kernel but found kernel size of shape {}".format(k))


class _LutronFilterBase(Module, _LutronFilter, ABC):
    def __init__(self, wrapped: Module, filter_parameters: FilterParameters):
        super().__init__()
        self._wrapped = wrapped
        self._bin = Binarize()
        self._filter_parameters = copy.copy(filter_parameters)

    @property
    def filter_parameters(self) -> FilterParameters:
        return self._filter_parameters

    def forward(self, x: Tensor) -> Tensor:
        return self._bin(self._wrapped(x))

    def generate_io_tensors(self) -> tuple[Tensor, Tensor]:
        kernel_size = _unwrap_1d_kernel_size(self.filter_parameters.kernel_size)

        inputs = generate_input_tensor(
            kernel_size=kernel_size,
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
