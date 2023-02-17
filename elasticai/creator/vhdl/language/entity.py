from abc import ABC, abstractmethod

from elasticai.creator.vhdl.language.ports import Port


class Entity(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def port(self) -> Port:
        ...

    @property
    @abstractmethod
    def lines(self) -> list[str]:
        ...
