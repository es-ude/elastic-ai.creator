from typing import Protocol

from elasticai.creator.mlframework import Module
from elasticai.creator.vhdl.designs.vhdl_design import VHDLDesign


class HWEquivalentLayer(VHDLDesign, Module, Protocol):
    ...
