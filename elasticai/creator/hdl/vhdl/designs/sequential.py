from abc import ABC, abstractmethod
from copy import copy
from dataclasses import dataclass
from enum import Enum
from functools import partial
from itertools import chain
from typing import Iterator

from elasticai.creator.hdl.code_generation.abstract_base_template import (
    module_to_package,
)
from elasticai.creator.hdl.code_generation.code_generation import (
    calculate_address_width,
)
from elasticai.creator.hdl.design_base.design import Design, Port
from elasticai.creator.hdl.design_base.std_signals import (
    clock,
    done,
    enable,
    x,
    x_address,
    y,
    y_address,
)
from elasticai.creator.hdl.translatable import Path
from elasticai.creator.hdl.vhdl.code_generation import (
    create_instance,
    signal_definition,
)
from elasticai.creator.hdl.vhdl.code_generation.code_generation import (
    create_connections,
    create_signal_definitions,
)
from elasticai.creator.hdl.vhdl.code_generation.template import Template


class Sequential(Design):
    def __init__(
        self,
        sub_designs: list[Design],
        *,
        x_width: int,
        y_width: int,
    ):
        super().__init__("Sequential")
        self._x_width = x_width
        self._y_width = y_width
        self.instances: dict[str, Design] = {}
        self._names: dict[str, int] = {}
        for design in sub_designs:
            self._register_subdesign(design)

    @property
    def _y_address_width(self) -> int:
        return calculate_address_width(self._y_width)

    @property
    def _x_address_width(self) -> int:
        return calculate_address_width(self._x_width)

    def _make_name_unique(self, name: str) -> str:
        return f"i_{name}_{self._get_counter_for_name(name)}"

    def _get_counter_for_name(self, name: str) -> int:
        if name in self._names:
            return self._names[name]
        else:
            return 0

    def _increment_name_counter(self, name: str):
        self._names[name] = 1 + self._get_counter_for_name(name)

    def _register_subdesign(self, d: Design):
        unique_name = self._make_name_unique(d.name)
        self.instances[unique_name] = d
        self._increment_name_counter(d.name)

    def _qualified_signal_name(self, instance: str, signal: str) -> str:
        return f"{instance}_{signal}"

    def _default_port(self) -> Port:
        return Port(incoming=[x(1), y_address(1)], outgoing=[y(1), x_address(1)])

    def _get_width_of_signal_or_default(self, signal_name: str, default: int) -> int:
        return self._get_width_for_signal_from_designs_or_default(
            signal_name, designs=iter(self.instances.values()), default=default
        )

    @staticmethod
    def _get_width_for_signal_from_designs_or_default(
        selected_signal: str, default: int, designs: Iterator[Design]
    ) -> int:
        for sub_design in designs:
            if selected_signal in sub_design.port.signal_names:
                return sub_design.port[selected_signal].width
        return default

    def _reversed_get_width_of_signal_or_default(
        self, signal_name: str, default: int
    ) -> int:
        return self._get_width_for_signal_from_designs_or_default(
            signal_name, default, reversed(self.instances.values())
        )

    @property
    def port(self) -> Port:
        return Port(
            incoming=[
                x(width=self._x_width),
                y_address(width=self._y_address_width),
                clock(),
                enable(),
            ],
            outgoing=[
                y(width=self._y_width),
                x_address(width=self._x_address_width),
                done(),
            ],
        )

    def _save_subdesigns(self, destination: Path) -> None:
        for name, design in self.instances.items():
            design.save_to(destination.create_subpath(name))

    @staticmethod
    def _connection(a: str, b: str) -> str:
        return f"{a} <= {b};"

    def _create_dataflow_nodes(self) -> list["_BaseNode"]:
        nodes: list[_BaseNode] = [_StartNode()]
        for instance, design in self.instances.items():
            node: _BaseNode = _DataFlowNode(instance=instance)
            node.add_sinks([signal.name for signal in design.port.incoming])
            node.add_sources([signal.name for signal in design.port.outgoing])
            nodes.append(node)
        nodes.append(_EndNode())
        return nodes

    def _generate_connections(self) -> list[str]:
        nodes = self._create_dataflow_nodes()
        wirer = _AutoWirer(nodes=nodes)
        return create_connections(wirer.connect())

    def _generate_instantiations(self) -> list[str]:
        instantiations: list[str] = list()
        for instance, design in self.instances.items():
            signal_map = {
                signal.name: self._qualified_signal_name(instance, signal.name)
                for signal in design.port
            }
            instantiations.extend(
                create_instance(
                    name=instance,
                    entity=design.name,
                    library="work",
                    architecture="rtl",
                    signal_mapping=signal_map,
                )
            )
        return instantiations

    def _generate_signal_definitions(self) -> list[str]:
        return sorted(
            chain.from_iterable(
                create_signal_definitions(f"{instance_id}_", instance.port.signals)
                for instance_id, instance in self.instances.items()
            )
        )

    def save_to(self, destination: Path):
        network_implementation = Template(
            "network", package=module_to_package(self.__module__)
        )
        self._save_subdesigns(destination)
        network_implementation.update_parameters(
            layer_connections=self._generate_connections(),
            layer_instantiations=self._generate_instantiations(),
            signal_definitions=self._generate_signal_definitions(),
            x_address_width=str(self._x_address_width),
            y_address_width=str(self._y_address_width),
            x_width=str(self._x_width),
            y_width=str(self._y_width),
        )
        target_file = destination.as_file(".vhd")
        target_file.write_text(network_implementation.lines())


class _AutoWirer:
    def __init__(self, nodes: list["_BaseNode"]):
        self.nodes = nodes
        self.mapping = {
            "x": ["x", "y"],
            "y": ["y", "x"],
            "x_address": ["x_address", "y_address"],
            "y_address": ["y_address", "x_address"],
            "enable": ["enable", "done"],
            "done": ["done", "enable"],
            "clock": ["clock"],
        }
        self.available_sources: dict[str, "_Source"] = {}

    def _pick_best_matching_source(self, sink: "_Sink") -> "_Source":
        for source_name in self.mapping[sink.name]:
            if source_name in self.available_sources:
                source = self.available_sources[source_name]
                return source
        return _TopNode().sources[0]

    def _update_available_sources(self, node: "_BaseNode") -> None:
        self.available_sources.update({s.name: s for s in node.sources})

    def connect(self) -> dict[str, str]:
        connections: dict[str, str] = {}
        for node in self.nodes:
            for sink in node.sinks:
                source = self._pick_best_matching_source(sink)
                connections[sink.get_qualified_name()] = source.get_qualified_name()
            self._update_available_sources(node)

        return connections


class _BaseNode(ABC):
    def __init__(self) -> None:
        self.sinks: list[_Sink] = []
        self.sources: list[_Source] = []

    def add_sinks(self, sinks: list[str]):
        create_sink = partial(_Sink, owner=self)
        self.sinks.extend(map(create_sink, sinks))

    def add_sources(self, sources: list[str]):
        create_source = partial(_Source, owner=self)
        self.sources.extend(map(create_source, sources))

    @property
    @abstractmethod
    def prefix(self) -> str:
        ...


class _TopNode(_BaseNode):
    @property
    def prefix(self) -> str:
        return ""


class _StartNode(_TopNode):
    def __init__(self):
        super().__init__()
        self.add_sources(["x", "y_address", "enable", "clock"])

    @property
    def prefix(self) -> str:
        return ""


class _EndNode(_TopNode):
    def __init__(self):
        super().__init__()
        self.add_sinks(["y", "x_address", "done"])


class _DataFlowNode(_BaseNode):
    def __init__(self, instance: str):
        self.instance = instance
        super().__init__()

    @property
    def prefix(self) -> str:
        return f"{self.instance}_"


class _Sink:
    def __init__(self, name: str, owner: _DataFlowNode):
        self.name = name
        self.owner = owner
        self.source: _DataFlowNode | None = None

    def get_qualified_name(self) -> str:
        return f"{self.owner.prefix}{self.name}"


class _Source:
    def __init__(self, name: str, owner: _DataFlowNode):
        self.name = name
        self.owner = owner
        self.sinks: list[_DataFlowNode] = []

    def get_qualified_name(self) -> str:
        return f"{self.owner.prefix}{self.name}"
