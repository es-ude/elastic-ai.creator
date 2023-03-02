from typing import cast

from torch.nn import Module as _torchModule
from torch.nn import Sequential as torchSequential

from elasticai.creator.hdl.vhdl.designs.sequential import (
    Sequential as _SequentialDesign,
)

from ..designs.design import Design
from .module import Module


class Sequential(torchSequential):
    def __init__(self, submodules: tuple[Module]):
        super().__init__(*cast(tuple[_torchModule], submodules))

    def translate(self) -> Design:
        submodules: list[Module] = [cast(Module, m) for m in self.children()]
        subdesigns = [m.translate() for m in submodules]
        if len(subdesigns) == 0:
            x_width = 1
            y_width = 1
        else:
            front = subdesigns[0]
            back = subdesigns[-1]
            x_width = front.port["x"].width
            y_width = back.port["y"].width
        return _SequentialDesign(
            sub_designs=subdesigns, x_width=x_width, y_width=y_width
        )
