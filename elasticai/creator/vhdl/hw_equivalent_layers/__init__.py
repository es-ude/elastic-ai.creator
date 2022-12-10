from itertools import chain
from typing import Any

import torch.nn

from elasticai.creator.nn.hard_sigmoid import (
    FixedPointHardSigmoid as nnFixedPointHardSigmoid,
)
from elasticai.creator.nn.linear import FixedPointLinear as nnFixedPointLinear
from mlframework import Module
from vhdl.code import CodeModule, CodeModuleBase, Translatable, Code
from vhdl.code_files.utils import calculate_address_width
from vhdl.hw_equivalent_layers.hw_blocks import (
    BufferedBaseHWBlockInterface,
    BaseHWBlockInterfaceInterface,
    HWBlockInterface,
    BufferedHWBlockInterface,
)
from vhdl.hw_equivalent_layers.vhdl_files import VHDLFile
from vhdl.model_tracing import HWEquivalentTracer
from vhdl.number_representations import FixedPointFactory


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
        # noinspection PyTypeChecker
        module: Module = self
        graph = HWEquivalentTracer().trace(module)
        signals = chain.from_iterable(
            (
                node.hw_equivalent_layer.signals(node.name)
                for node in graph.hw_equivalent_nodes
            )
        )
        layer_instantiations = chain.from_iterable(
            (
                node.hw_equivalent_layer.instantiation(node.name)
                for node in graph.hw_equivalent_nodes
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


class FixedPointHardSigmoid(nnFixedPointHardSigmoid, HWBlockInterface):
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
        self._hw_block = BaseHWBlockInterfaceInterface(data_width=data_width)


class FixedPointLinear(nnFixedPointLinear, BufferedHWBlockInterface):
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
        self._hw_block = BufferedBaseHWBlockInterface(
            data_width,
            x_address_width=calculate_address_width(in_features),
            y_address_width=calculate_address_width(out_features),
        )
