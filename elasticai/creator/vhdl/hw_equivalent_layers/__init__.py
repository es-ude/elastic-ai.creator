from itertools import chain
from typing import Iterable

import torch.nn

from resource_utils import read_text
from vhdl.code import Code, CodeModule, CodeModuleBase, CodeFileBase
from vhdl.hw_blocks import BufferedBaseHWBlock, BaseHWBlock
from vhdl.templates.utils import expand_template, expand_multiline_template
from elasticai.creator.nn.linear import FixedPointLinear as _FixedPointLinear


class FixedPointLinear(_FixedPointLinear):
    pass


class RootModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.elasticai_tags = {
            "x_address_width": 1,
            "y_address_width": 1,
            "data_width": 1,
        }

    def signals(self, prefix: str) -> Code:
        return ""

    def instantiation(self, prefix: str) -> Code:
        return ""

    def data_width(self) -> int:
        ...

    def x_address_width(self) -> int:
        ...

    def y_address_width(self) -> int:
        ...

    def to_vhdl(self) -> Iterable[CodeModule]:
        code = read_text("elasticai.creator.vhdl.templates", "network.tpl.vhd")
        code = expand_template(code, **self.elasticai_tags)
        fp_linear_block = BufferedBaseHWBlock(
            x_address_width=self.elasticai_tags["x_address_width"],
            y_address_width=self.elasticai_tags["y_address_width"],
            data_width=self.elasticai_tags["data_width"],
        )
        sigmoid_block = BaseHWBlock(data_width=self.elasticai_tags["data_width"])
        signals = chain(
            fp_linear_block.signals("fp_linear"),
            sigmoid_block.signals("fp_hard_sigmoid"),
        )
        layer_instantiations = chain(
            fp_linear_block.instantiation("fp_linear"),
            sigmoid_block.instantiation("fp_hard_sigmoid"),
        )
        code = expand_multiline_template(
            code, signal_definitions=signals, layer_instantiations=layer_instantiations
        )

        yield from [
            CodeModuleBase(
                name="network",
                files=(CodeFileBase(name="network.vhd", code=code),),
            )
        ]
