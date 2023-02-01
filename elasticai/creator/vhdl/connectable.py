from abc import abstractmethod
from typing import Protocol, TypeVar

T_Connectable_contra = TypeVar(
    "T_Connectable_contra", bound="Connectable", contravariant=True
)
T_contra = TypeVar("T_contra", contravariant=True)


class Connectable(Protocol[T_contra]):
    @abstractmethod
    def connect(self: T_Connectable_contra, other: T_contra):
        ...

    @abstractmethod
    def is_missing_inputs(self) -> bool:
        ...
