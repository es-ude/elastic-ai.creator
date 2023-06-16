from abc import ABC, abstractmethod

from elasticai.creator.vhdl.design_base.ports import Port
from elasticai.creator.vhdl.savable import Path, Savable


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
