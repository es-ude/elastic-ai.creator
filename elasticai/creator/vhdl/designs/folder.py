from abc import abstractmethod
from typing import Iterable, Protocol


class Folder(Protocol):
    @abstractmethod
    def new_file(self, name: str, content: Iterable[str]) -> None:
        """Throws exception if file already exists"""
        ...

    @abstractmethod
    def new_folder(self, name: str) -> "Folder":
        """Idempotent"""
        ...
