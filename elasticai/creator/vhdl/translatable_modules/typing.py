from abc import abstractmethod
from typing import Protocol

from elasticai.creator.mlframework import Module


class HWEquivalentLayer(Module, Protocol):
    ...


class Node(Protocol):
    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def next(self) -> "Node":
        ...

    @property
    @abstractmethod
    def op(self) -> str:
        ...
