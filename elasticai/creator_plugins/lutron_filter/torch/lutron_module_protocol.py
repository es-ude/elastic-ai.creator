from abc import abstractmethod
from typing import Any, Protocol, runtime_checkable

from torch import Tensor

from elasticai.creator.ir.helpers import FilterParameters


@runtime_checkable
class LutronModule(Protocol):
    @abstractmethod
    def forward(self, x: Tensor) -> Tensor: ...

    @property
    @abstractmethod
    def creator_meta(self) -> dict[str, Any]: ...

    @property
    @abstractmethod
    def filter_parameters(self) -> FilterParameters: ...

    @abstractmethod
    def generate_io_tensors(self) -> tuple[Tensor, Tensor]: ...

    @abstractmethod
    def infer_shape(
        self,
        input_shape: tuple[int, int, int],
        output_shape: tuple[int, int, int],
    ) -> None: ...
