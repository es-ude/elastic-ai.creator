from abc import abstractmethod
from pathlib import Path
from typing import Protocol


class Savable(Protocol):
    @abstractmethod
    def save_to(self, destination: Path) -> None:
        ...
