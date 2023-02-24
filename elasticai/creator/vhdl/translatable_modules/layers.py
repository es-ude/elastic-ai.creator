import dataclasses
import typing
from typing import Any, Collection

import torch.nn

import elasticai.creator.mlframework
from elasticai.creator.nn.hard_sigmoid import HardSigmoid as nnHardSigmoid
from elasticai.creator.nn.linear import FixedPointLinear as nnFixedPointLinear
from elasticai.creator.vhdl.code_files.utils import calculate_address_width
from elasticai.creator.vhdl.designs._network_blocks import (
    BufferedNetworkBlock,
    NetworkBlock,
)
from elasticai.creator.vhdl.designs.folder import Folder
from elasticai.creator.vhdl.designs.vhdl_code_generation import (
    create_instance,
    signal_definition,
)
from elasticai.creator.vhdl.hardware_description_language.design import Design, Instance
from elasticai.creator.vhdl.number_representations import FixedPointConfig
from elasticai.creator.vhdl.templates import VHDLTemplate
from elasticai.creator.vhdl.tracing.hw_equivalent_fx_tracer import HWEquivalentFXTracer
from elasticai.creator.vhdl.tracing.typing import HWEquivalentGraph, TranslatableLayer


class RootModule(torch.nn.Module, elasticai.creator.mlframework.Module):
    def __init__(self):
        self.elasticai_tags = {
            "x_address_width": 1,
            "y_address_width": 1,
            "x_width": 1,
            "y_width": 1,
        }
        torch.nn.Module.__init__(self)

    def _template_parameters(self) -> dict[str, str]:
        return dict(((key, str(value)) for key, value in self.elasticai_tags.items()))

    def _get_tracer(self) -> HWEquivalentFXTracer:
        return HWEquivalentFXTracer()

    def translate(self):
        graph = self._get_tracer().trace(self)

        def collect_instances():
            instances = []
            for node in graph.module_nodes:
                module = graph.get_module_for_node(node)
                design = module.translate()
                instance = design.instantiate(node.name)
                instances.append(instance)

            return instances

        instances = collect_instances()
        nested_signals = (instance.design.port.signals for instance in instances)
        flattened_signals = [signal for signals in nested_signals for signal in signals]
        signal_definitions = [
            signal_definition(name=s.id, width=s.width) for s in flattened_signals
        ]

        def instance_to_code(instance: Instance) -> str:
            signal_names = (s.id() for s in instance.design.port.signals)
            signal_map = dict((s, f"{instance.name}_{s}") for s in signal_names)
            return create_instance(
                entity=instance.design.name,
                name=instance.name,
                signal_mapping=signal_map,
                library="work",
            )

        layer_instantiations = [instance_to_code(i) for i in instances]
        layer_connections = []
        file = VHDLTemplate(
            base_name="network",
            signals=signal_definitions,
            layer_instantiations=layer_instantiations,
            layer_connections=layer_connections,
        )
        file.update_parameters(**self._template_parameters())
        design = CodeModuleBase(name="network", files=(file,))
        return design


class FPHardSigmoidBlock(NetworkBlock):
    def __init__(self, data_width):
        super().__init__("", x_width=data_width, y_width=data_width)

    def save_to(self, destination: Folder):
        ...


class FixedPointHardSigmoid(nnHardSigmoid):
    def __init__(
        self,
        in_place: bool = False,
        *,
        data_width,
    ):
        super().__init__(in_place)
        self._hw_block = FPHardSigmoidBlock(data_width)

    def translate(self) -> Design:
        return self._hw_block


T = typing.TypeVar("T")


class FixedPointLinearBlock(BufferedNetworkBlock):
    def save_to(self, destination: Folder):
        ...


class FixedPointLinear(nnFixedPointLinear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        fixed_point_config: FixedPointConfig,
        bias: bool = True,
        device: Any = None,
        *,
        data_width: int,
    ):
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            fixed_point_factory=fixed_point_config,
            bias=bias,
            device=device,
        )
        self._hw_block = FixedPointLinearBlock(
            name="fp_linear",
            y_width=data_width,
            x_width=data_width,
            x_address_width=calculate_address_width(in_features),
            y_address_width=calculate_address_width(out_features),
        )

    def translate(self):
        return self._hw_block


@dataclasses.dataclass
class CodeModuleBase:
    @property
    def submodules(self) -> Collection:
        return self._submodules

    @property
    def files(self) -> Collection:
        return self._files

    @property
    def name(self) -> str:
        return self._name

    def __init__(
        self,
        name: str,
        files: Collection,
        submodules: Collection = tuple(),
    ):
        self._name = name
        self._files = files
        self._submodules = submodules
