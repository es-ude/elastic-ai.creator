from abc import ABC, abstractmethod
from pathlib import Path

from elasticai.creator.file_generation.v2.savable import Savable
from elasticai.creator.vhdl.design.ports import Port


class Design(Savable, ABC):
    def __init__(self, name: str):
        self.name = name

    @property
    @abstractmethod
    def port(self) -> Port:
        ...

    @abstractmethod
    def save_to(self, destination: Path) -> None:
        ...
