import typing
from abc import ABC, abstractmethod
from typing import Any

import torch.nn

from elasticai.creator.mlframework import Module
from elasticai.creator.nn.hard_sigmoid import HardSigmoid as nnHardSigmoid
from elasticai.creator.nn.linear import FixedPointLinear as nnFixedPointLinear
from elasticai.creator.vhdl.code import Code, CodeModule, CodeModuleBase, Translatable
from elasticai.creator.vhdl.code_files.utils import calculate_address_width
from elasticai.creator.vhdl.hw_equivalent_layers.hw_blocks import (
    BaseHWBlock,
    BufferedBaseHWBlock,
)
from elasticai.creator.vhdl.hw_equivalent_layers.tracing import (
    HWEquivalentFXTracer,
    HWEquivalentTracer,
    create_hw_block_collection,
)
from elasticai.creator.vhdl.number_representations import FixedPointConfig
from elasticai.creator.vhdl.vhdl_files import VHDLFile


class AbstractTranslatableLayer(Translatable, ABC):
    def __init__(self, hw_block: BaseHWBlock):
        self._hw_block = hw_block

    @abstractmethod
    def _get_tracer(self) -> HWEquivalentTracer:
        ...

    @abstractmethod
    def _template_parameters(self) -> dict[str, str]:
        return {
            name: str(getattr(self, name))
            for name in ("x_width", "y_width", "x_address_width", "y_address_width")
        }

    def signal_definitions(self, prefix: str) -> Code:
        return self._hw_block.get_port().build_port_map(prefix).signal_definitions()

    def instantiation(self, prefix: str) -> Code:
        return self._hw_block.get_port().build_port_map(prefix).code()

    def translate(self) -> CodeModule:
        module: Module = typing.cast(Module, self)
        graph = self._get_tracer().trace(module)
        blocks = create_hw_block_collection(graph)
        signals = blocks.signals("")
        layer_instantiations = list(blocks.instantiations(""))
        layer_connections = ["fp_linear_x <= x;"]
        return CodeModuleBase(
            name="network",
            files=(
                VHDLFile(
                    name="network",
                    signal_definitions=signals,
                    layer_instantiations=layer_instantiations,
                    layer_connections=layer_connections,
                    **self._template_parameters(),
                ),
            ),
        )


class RootModule(torch.nn.Module, AbstractTranslatableLayer):
    def __init__(self):
        self.elasticai_tags = {
            "x_address_width": 1,
            "y_address_width": 1,
            "x_width": 1,
            "y_width": 1,
        }
        torch.nn.Module.__init__(self)
        AbstractTranslatableLayer.__init__(
            self, BufferedBaseHWBlock(**self.elasticai_tags)
        )

    def _template_parameters(self) -> dict[str, str]:
        return dict(((key, str(value)) for key, value in self.elasticai_tags.items()))

    def translate(self):
        return super().translate()

    def _get_tracer(self) -> HWEquivalentTracer:
        return HWEquivalentFXTracer()


class FixedPointHardSigmoid(nnHardSigmoid):
    def signal_definitions(self, prefix: str) -> Code:
        return self._hw_block.get_port().build_port_map(prefix).signal_definitions()

    def instantiation(self, prefix: str) -> Code:
        return self._hw_block.get_port().build_port_map(prefix).code()

    def __init__(
        self,
        fixed_point_factory: FixedPointConfig,
        in_place: bool = False,
        *,
        data_width,
    ):
        super().__init__(in_place)
        self._hw_block = BaseHWBlock(x_width=data_width, y_width=data_width)


class FixedPointLinear(nnFixedPointLinear):
    def signal_definitions(self, prefix: str) -> Code:
        return self._hw_block.get_port().build_port_map(prefix).signal_definitions()

    def instantiation(self, prefix: str) -> Code:
        return self._hw_block.get_port().build_port_map(prefix).code()

    def __init__(
        self,
        in_features: int,
        out_features: int,
        fixed_point_factory: FixedPointConfig,
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
            y_width=data_width,
            x_width=data_width,
            x_address_width=calculate_address_width(in_features),
            y_address_width=calculate_address_width(out_features),
        )
