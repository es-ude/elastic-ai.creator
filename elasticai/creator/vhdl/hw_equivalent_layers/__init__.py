import typing
from abc import abstractmethod, ABC
from typing import Any

import torch.nn

from elasticai.creator.mlframework import Module
from elasticai.creator.nn.hard_sigmoid import (
    FixedPointHardSigmoid as nnFixedPointHardSigmoid,
)
from elasticai.creator.nn.linear import FixedPointLinear as nnFixedPointLinear
from elasticai.creator.vhdl.code import CodeModule, CodeModuleBase, Translatable, Code
from elasticai.creator.vhdl.code_files.utils import calculate_address_width
from elasticai.creator.vhdl.hw_equivalent_layers.hw_blocks import (
    BufferedBaseHWBlock,
    BaseHWBlock,
    HWBlockInterface,
    BufferedHWBlockInterface,
)
from elasticai.creator.vhdl.hw_equivalent_layers.vhdl_files import VHDLFile
from elasticai.creator.vhdl.model_tracing import (
    HWEquivalentTracer,
    create_hw_block_collection,
    Tracer,
)
from elasticai.creator.vhdl.number_representations import FixedPointFactory


class AbstractTranslatableLayer(
    torch.nn.Module, BufferedBaseHWBlock, Translatable, ABC
):
    @abstractmethod
    def _get_tracer(self) -> Tracer:
        ...

    def _stringify_tags(self) -> dict[str, str]:
        return dict(((k, str(v)) for k, v in self.elasticai_tags.items()))

    def translate(self) -> CodeModule:
        module: Module = typing.cast(Module, self)
        graph = self._get_tracer().trace(module)
        blocks = create_hw_block_collection(graph)
        signals = blocks.signals("")
        layer_instantiations = blocks.instantiations("")
        layer_connections = ["fp_linear_x <= x;"]
        return CodeModuleBase(
            name="network",
            files=(
                VHDLFile(
                    name="network",
                    signal_definitions=signals,
                    layer_instantiations=layer_instantiations,
                    layer_connections=layer_connections,
                    **self._stringify_tags(),
                ),
            ),
        )


class RootModule(AbstractTranslatableLayer):
    def __init__(self):
        super().__init__()
        self.elasticai_tags = {
            "x_address_width": 1,
            "y_address_width": 1,
            "x_width": 1,
            "y_width": 1,
        }

    def _get_tracer(self) -> Tracer:
        return HWEquivalentTracer()


class FixedPointHardSigmoid(nnFixedPointHardSigmoid, HWBlockInterface):
    @property
    def x_width(self) -> int:
        return self._hw_block.x_width

    @property
    def y_width(self) -> int:
        return self._hw_block.y_width

    def signal_definitions(self, prefix: str) -> Code:
        return self._hw_block.signal_definitions(prefix)

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
        self._hw_block = BaseHWBlock(x_width=data_width, y_width=data_width)


class FixedPointLinear(nnFixedPointLinear, BufferedHWBlockInterface):
    @property
    def y_address_width(self) -> int:
        return self._hw_block.y_address_width

    @property
    def x_width(self) -> int:
        return self._hw_block.x_width

    @property
    def y_width(self) -> int:
        return self._hw_block.y_width

    @property
    def x_address_width(self) -> int:
        return self._hw_block.x_address_width

    def signal_definitions(self, prefix: str) -> Code:
        return self._hw_block.signal_definitions(prefix)

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
            y_width=data_width,
            x_width=data_width,
            x_address_width=calculate_address_width(in_features),
            y_address_width=calculate_address_width(out_features),
        )
