from abc import abstractmethod
from typing import Protocol

from elasticai.creator.hdl.design_base.design import Design
from elasticai.creator.nn.module import Module as _BaseModule


class Tensor(Protocol):
    ...


class Module(_BaseModule[Design], Protocol):
    @abstractmethod
    def translate(self) -> Design:
        ...
