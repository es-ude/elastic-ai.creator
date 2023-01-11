from abc import abstractmethod
from typing import Any, Protocol, TypeVar

T_Connectable_contra = TypeVar(
    "T_Connectable_contra", bound="Connectable", contravariant=True
)


class Connectable(Protocol):
    @abstractmethod
    def connect(self: T_Connectable_contra, other: Any):
        ...

    @abstractmethod
    def is_missing_inputs(self) -> bool:
        ...
