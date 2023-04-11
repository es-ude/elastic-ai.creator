from abc import abstractmethod
from typing import Protocol

from elasticai.creator.hdl.translatable import Saveable
from elasticai.creator.nn.module import Module as _BaseModule


class Tensor(Protocol):
    ...


class Module(_BaseModule[Saveable], Protocol):
    @abstractmethod
    def translate(self) -> Saveable:
        ...
