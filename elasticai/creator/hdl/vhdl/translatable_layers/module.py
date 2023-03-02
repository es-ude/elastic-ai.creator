from abc import abstractmethod
from typing import Protocol

from elasticai.creator.hdl.vhdl.designs.design import Design


class Tensor(Protocol):
    ...


class Module(Protocol):
    def __call__(self, x) -> Tensor:
        ...

    def eval(self) -> None:
        ...

    def train(self) -> None:
        ...

    @abstractmethod
    def translate(self) -> Design:
        ...
