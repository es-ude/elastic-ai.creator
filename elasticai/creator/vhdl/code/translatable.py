from abc import abstractmethod
from typing import Protocol, TypeVar

T_co = TypeVar("T_co", covariant=True)


class Translatable(Protocol[T_co]):
    @abstractmethod
    def translate(self) -> T_co:
        ...
