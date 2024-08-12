from abc import ABC, abstractmethod
from pathlib import Path


class Design(ABC):
    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def save_to(self, destination: Path) -> None:
        ...
