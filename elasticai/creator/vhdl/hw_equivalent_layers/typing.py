from typing import Protocol

from mlframework import Module
from vhdl.code import Translatable
from vhdl.hw_equivalent_layers import HWBlockInterface, BufferedHWBlockInterface


class HWEquivalentLayer(HWBlockInterface, Translatable, Module, Protocol):
    ...


class BufferedHWEquivalentLayer(
    BufferedHWBlockInterface, Module, Translatable, Protocol
):
    ...
