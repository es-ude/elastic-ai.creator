from abc import ABC, abstractmethod

from elasticai.creator.hdl.design_base.ports import Port
from elasticai.creator.hdl.savable import Path, Savable


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
