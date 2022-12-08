from abc import abstractmethod
from typing import Protocol

from vhdl.code import Code, Translatable
from vhdl.components.network_component import (
    SignalsForBufferlessComponent,
    ComponentInstantiation,
    SignalsForComponentWithBuffer,
    BufferedComponentInstantiation,
)


class HWBlock(Protocol):
    @property
    @abstractmethod
    def data_width(self) -> int:
        ...

    @abstractmethod
    def signals(self, prefix: str) -> Code:
        ...

    @abstractmethod
    def instantiation(self, prefix: str) -> Code:
        ...


class BufferedHWBlock(HWBlock, Protocol):
    @property
    @abstractmethod
    def x_address_width(self) -> int:
        ...

    @property
    @abstractmethod
    def y_address_width(self) -> int:
        ...


class TranslatableHWBlock(HWBlock, Translatable, Protocol):
    ...


class TranslatableBufferedHWBlock(BufferedHWBlock, Translatable, Protocol):
    ...


class BaseHWBlock(HWBlock):
    def __init__(self, data_width: int):
        self._data_width = data_width

    @property
    def data_width(self) -> int:
        return self._data_width

    def signals(self, prefix: str) -> Code:
        yield from SignalsForBufferlessComponent(
            name=prefix, data_width=self.data_width
        ).code()

    def instantiation(self, prefix: str) -> Code:
        yield from ComponentInstantiation(name=prefix).code()


class BufferedBaseHWBlock(BufferedHWBlock):
    def __init__(self, data_width: int, x_address_width: int, y_address_width):
        self._data_width = data_width
        self._x_address_width = x_address_width
        self._y_address_width = y_address_width

    @property
    def data_width(self) -> int:
        return self._data_width

    @property
    def x_address_width(self) -> int:
        return self._x_address_width

    @property
    def y_address_width(self) -> int:
        return self._y_address_width

    def signals(self, prefix: str) -> Code:
        yield from SignalsForComponentWithBuffer(
            data_width=self.data_width,
            name=prefix,
            x_address_width=self.x_address_width,
            y_address_width=self.y_address_width,
        ).code()

    def instantiation(self, prefix: str) -> Code:
        yield from BufferedComponentInstantiation(name=prefix).code()
