from itertools import chain
from typing import Any

import torch.nn

from elasticai.creator.nn.linear import FixedPointLinear as nnFixedPointLinear
from vhdl.code import CodeModule, CodeModuleBase, Translatable, Code
from vhdl.code_files.utils import calculate_address_width
from vhdl.hw_blocks import BufferedBaseHWBlock, BaseHWBlock, HWBlock, BufferedHWBlock
from vhdl.number_representations import FixedPointFactory
from vhdl.vhdl_files import VHDLFile
from elasticai.creator.nn.hard_sigmoid import (
    FixedPointHardSigmoid as nnFixedPointHardSigmoid,
)


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
        signals = list(
            chain(
                self.fp_linear.signals("fp_linear"),
                self.fp_hard_sigmoid.signals("fp_hard_sigmoid"),
            )
        )
        layer_instantiations = list(
            chain(
                self.fp_linear.instantiation("fp_linear"),
                self.fp_hard_sigmoid.instantiation("fp_hard_sigmoid"),
            )
        )
        return CodeModuleBase(
            name="network",
            files=(
                VHDLFile(
                    name="network",
                    signal_definitions=signals,
                    layer_instantiations=layer_instantiations,
                    **self._stringify_tags(),
                ),
            ),
        )


class FixedPointHardSigmoid(nnFixedPointHardSigmoid, HWBlock):
    @property
    def data_width(self) -> int:
        return self._hw_block.data_width

    def signals(self, prefix: str) -> Code:
        return self._hw_block.signals(prefix)

    def instantiation(self, prefix: str) -> Code:
        return self._hw_block.instantiation(prefix)

    def __init__(
        self,
        fixed_point_factory: FixedPointFactory,
        in_place: bool = False,
        *,
        data_width,
    ):
        super().__init__(fixed_point_factory, in_place)
        self._hw_block = BaseHWBlock(data_width=data_width)


class FixedPointLinear(nnFixedPointLinear, BufferedHWBlock):
    @property
    def x_address_width(self) -> int:
        return self._hw_block.x_address_width

    @property
    def y_address_width(self) -> int:
        return self._hw_block.x_address_width

    @property
    def data_width(self) -> int:
        return self._hw_block.data_width

    def signals(self, prefix: str) -> Code:
        return self._hw_block.signals(prefix)

    def instantiation(self, prefix: str) -> Code:
        return self._hw_block.instantiation(prefix)

    def __init__(
        self,
        in_features: int,
        out_features: int,
        fixed_point_factory: FixedPointFactory,
        bias: bool = True,
        device: Any = None,
        *,
        data_width: int,
    ):
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            fixed_point_factory=fixed_point_factory,
            bias=bias,
            device=device,
        )
        self._hw_block = BufferedBaseHWBlock(
            data_width,
            x_address_width=calculate_address_width(in_features),
            y_address_width=calculate_address_width(out_features),
        )
