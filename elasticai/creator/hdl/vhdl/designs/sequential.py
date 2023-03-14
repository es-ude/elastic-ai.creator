from abc import ABC, abstractmethod
from functools import partial
from itertools import chain

from elasticai.creator.hdl.code_generation.abstract_base_template import (
    module_to_package,
)
from elasticai.creator.hdl.code_generation.code_generation import (
    calculate_address_width,
)
from elasticai.creator.hdl.design_base import std_signals
from elasticai.creator.hdl.design_base.design import Design, Port
from elasticai.creator.hdl.design_base.signal import Signal
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
from elasticai.creator.hdl.vhdl.code_generation import create_instance
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
        x_address_width: int,
        y_address_width: int,
    ):
        super().__init__("Sequential")
        self._x_width = x_width
        self._y_width = y_width
        self._y_address_width = y_address_width
        self._x_address_width = x_address_width
        self._instances: dict[str, Design] = {}
        self._names: dict[str, int] = {}
        self._library_name_for_instances = "work"
        self._architecture_name_for_instances = "rtl"
        self._autowirer = _AutoWirer
        for design in sub_designs:
            self._register_subdesign(design)

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
        self._instances[unique_name] = d
        self._increment_name_counter(d.name)

    def _qualified_signal_name(self, instance: str, signal: str) -> str:
        return f"{instance}_{signal}"

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
        for name, design in self._instances.items():
            design.save_to(destination.create_subpath(name))

    def _create_dataflow_nodes(self) -> list["_DataFlowNode"]:
        nodes: list[_DataFlowNode] = [
            _StartNode(x_width=self._x_width, y_address_width=self._y_address_width)
        ]
        for instance, design in self._instances.items():
            nodes.append(self._create_data_flow_node(instance, design))
        nodes.append(
            _EndNode(y_width=self._y_width, x_address_width=self._x_address_width)
        )
        return nodes

    @staticmethod
    def _create_data_flow_node(instance: str, design: Design) -> "_DataFlowNode":
        node = _InstanceNode(instance=instance)
        node.add_sinks(design.port.incoming)
        node.add_sources(design.port.outgoing)
        return node

    def _generate_connections(self) -> list[str]:
        nodes = self._create_dataflow_nodes()
        wirer = self._autowirer(nodes=nodes)
        return create_connections(wirer.connect())

    def _generate_instantiations(self) -> list[str]:
        instantiations: list[str] = list()
        for instance, design in self._instances.items():
            signal_map = {
                signal.name: self._qualified_signal_name(instance, signal.name)
                for signal in design.port
            }
            instantiations.extend(
                create_instance(
                    name=instance,
                    entity=design.name,
                    library=self._library_name_for_instances,
                    architecture=self._architecture_name_for_instances,
                    signal_mapping=signal_map,
                )
            )
        return instantiations

    def _generate_signal_definitions(self) -> list[str]:
        return sorted(
            chain.from_iterable(
                create_signal_definitions(f"{instance_id}_", instance.port.signals)
                for instance_id, instance in self._instances.items()
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
        target_file = destination.create_subpath("sequential").as_file(".vhd")
        target_file.write_text(network_implementation.lines())


class _AutoWirer:
    def __init__(self, nodes: list["_DataFlowNode"]):
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
        self.available_sources: dict[str, "_OwnedSignal"] = {}

    def _pick_best_matching_source(self, sink: "_OwnedSignal") -> "_OwnedSignal":
        for source_name in self.mapping[sink.name]:
            if source_name in self.available_sources:
                source = self.available_sources[source_name]
                return source
        return _TopNode().sources[0]

    def _update_available_sources(self, node: "_DataFlowNode") -> None:
        self.available_sources.update({s.name: s for s in node.sources})

    def connect(self) -> dict[str, str]:
        connections: dict[str, str] = {}
        for node in self.nodes:
            for sink in node.sinks:
                source = self._pick_best_matching_source(sink)
                connections[sink.qualified_name] = source.qualified_name
            self._update_available_sources(node)

        return connections


class _DataFlowNode(ABC):
    def __init__(self) -> None:
        self.sinks: list[_OwnedSignal] = []
        self.sources: list[_OwnedSignal] = []

    def add_sinks(self, sinks: list[Signal]):
        create_sink = partial(_OwnedSignal, owner=self)
        self.sinks.extend(map(create_sink, sinks))

    def add_sources(self, sources: list[Signal]):
        create_source = partial(_OwnedSignal, owner=self)
        self.sources.extend(map(create_source, sources))

    @property
    @abstractmethod
    def prefix(self) -> str:
        ...


class _TopNode(_DataFlowNode):
    @property
    def prefix(self) -> str:
        return ""


class _StartNode(_TopNode):
    def __init__(self, x_width: int, y_address_width: int):
        super().__init__()
        self.add_sources(
            [
                std_signals.x(x_width),
                std_signals.y_address(y_address_width),
                std_signals.enable(),
                std_signals.clock(),
            ]
        )

    @property
    def prefix(self) -> str:
        return ""


class _EndNode(_TopNode):
    def __init__(self, y_width: int, x_address_width: int):
        super().__init__()
        self.add_sinks(
            [
                std_signals.y(y_width),
                std_signals.x_address(x_address_width),
                std_signals.done(),
            ]
        )


class _InstanceNode(_DataFlowNode):
    def __init__(self, instance: str):
        self.instance = instance
        super().__init__()

    @property
    def prefix(self) -> str:
        return f"{self.instance}_"


class _OwnedSignal:
    def __init__(self, signal: Signal, owner: _DataFlowNode):
        self.owner = owner
        self._signal = signal

    @property
    def name(self) -> str:
        return self._signal.name

    @property
    def width(self) -> int:
        return self._signal.width

    @property
    def qualified_name(self) -> str:
        return f"{self.owner.prefix}{self.name}"
