from typing import cast

from torch.nn import Module as _torchModule
from torch.nn import Sequential as torchSequential

from elasticai.creator.hdl.design_base.design import Design
from elasticai.creator.hdl.vhdl.designs.sequential import (
    Sequential as _SequentialDesign,
)

from .module import Module


class Sequential(torchSequential):
    def __init__(self, submodules: tuple[Module, ...]):
        super().__init__(*cast(tuple[_torchModule], submodules))

    def translate(self) -> Design:
        submodules: list[Module] = [cast(Module, m) for m in self.children()]
        subdesigns = [m.translate() for m in submodules]
        x_address_width = 1
        y_address_width = 1
        if len(subdesigns) == 0:
            x_width = 1
            y_width = 1

        else:
            front = subdesigns[0]
            back = subdesigns[-1]
            x_width = front.port["x"].width
            y_width = back.port["y"].width
            found_y_address = False
            found_x_address = False
            for design in subdesigns:
                if "y_address" in design.port and not found_y_address:
                    found_y_address = True
                    y_address_width = design.port["y_address"].width
                if "x_address" in design.port and not found_x_address:
                    found_x_address = True
                    x_address_width = back.port["x_address"].width
        return _SequentialDesign(
            sub_designs=subdesigns,
            x_width=x_width,
            y_width=y_width,
            x_address_width=x_address_width,
            y_address_width=y_address_width,
        )
