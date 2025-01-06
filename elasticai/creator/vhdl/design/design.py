from abc import ABC, abstractmethod

from elasticai.creator.file_generation.savable import Path, Savable
from elasticai.creator.vhdl.design.ports import Port


class Design(Savable, ABC):
    def __init__(self, name: str):
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    @abstractmethod
    def port(self) -> Port: ...

    @abstractmethod
    def save_to(self, destination: Path) -> None: ...
