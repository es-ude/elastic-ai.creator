from abc import ABC, abstractmethod
from typing import Iterable, Protocol


class Saveable(ABC):
    @abstractmethod
    def save_to(self, destination: "Path"):
        ...


class Path(Protocol):
    @abstractmethod
    def as_file(self, suffix: str) -> "File":
        ...

    @abstractmethod
    def create_subpath(self, subpath_name: str) -> "Path":
        ...


class File(Protocol):
    @abstractmethod
    def write_text(self, text: Iterable[str]) -> None:
        ...
