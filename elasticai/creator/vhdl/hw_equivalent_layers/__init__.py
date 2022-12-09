from itertools import chain

import torch.nn

from elasticai.creator.nn.linear import FixedPointLinear as _FixedPointLinear
from vhdl.code import CodeModule, CodeModuleBase, Translatable
from vhdl.hw_blocks import BufferedBaseHWBlock, BaseHWBlock
from vhdl.vhdl_files import VHDLFile


class FixedPointLinear(_FixedPointLinear):
    pass


class RootModule(torch.nn.Module, Translatable):
    def __init__(self):
        super().__init__()
        self.elasticai_tags = {
            "x_address_width": 1,
            "y_address_width": 1,
            "data_width": 1,
        }

    def _stringify_tags(self) -> dict[str, str]:
        return dict(((k, str(v)) for k, v in self.elasticai_tags.items()))

    def translate(self) -> CodeModule:
        fp_linear_block = BufferedBaseHWBlock(
            x_address_width=self.elasticai_tags["x_address_width"],
            y_address_width=self.elasticai_tags["y_address_width"],
            data_width=self.elasticai_tags["data_width"],
        )
        sigmoid_block = BaseHWBlock(data_width=self.elasticai_tags["data_width"])
        signals = list(
            chain(
                fp_linear_block.signals("fp_linear"),
                sigmoid_block.signals("fp_hard_sigmoid"),
            )
        )
        layer_instantiations = list(
            chain(
                fp_linear_block.instantiation("fp_linear"),
                sigmoid_block.instantiation("fp_hard_sigmoid"),
            )
        )
        return CodeModuleBase(
            name="network",
            files=(
                VHDLFile(
                    name="network",
                    signal_definitions=signals,
                    layer_instantiations=layer_instantiations,
                    **self._stringify_tags()
                ),
            ),
        )
