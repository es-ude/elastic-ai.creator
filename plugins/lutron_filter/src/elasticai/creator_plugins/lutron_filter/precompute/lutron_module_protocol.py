from abc import abstractmethod
from typing import Protocol, runtime_checkable

from torch import Tensor

from elasticai.creator_plugins.grouped_filter import FilterParameters


@runtime_checkable
class LutronModule(Protocol):
    @abstractmethod
    def forward(self, x: Tensor) -> Tensor: ...

    @abstractmethod
    def generate_io_tensors(self) -> tuple[Tensor, Tensor]: ...

    @abstractmethod
    def infer_shape(
        self,
        input_shape: tuple[int, int, int],
        output_shape: tuple[int, int, int],
    ) -> None: ...


class LutronFilter(LutronModule, Protocol):
    @property
    def filter_parameters(self) -> FilterParameters: ...
