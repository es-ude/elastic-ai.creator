from abc import abstractmethod
from typing import Iterable, Protocol


class File(Protocol):
    @abstractmethod
    def write_text(self, text: Iterable[str] | str) -> None:
        ...
