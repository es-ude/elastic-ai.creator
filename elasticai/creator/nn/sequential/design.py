from functools import partial, reduce
from itertools import chain

from elasticai.creator.vhdl.auto_wire_protocols.autowiring import (
    AutoWirer,
    DataFlowNode,
)
from elasticai.creator.vhdl.auto_wire_protocols.port_definitions import create_port
from elasticai.creator.vhdl.code_generation.code_abstractions import (
    create_connections_using_to_from_pairs,
    create_instance,
    create_signal_definitions,
)
from elasticai.creator.vhdl.code_generation.template import (
    InProjectTemplate,
    module_to_package,
)
from elasticai.creator.vhdl.design_base.design import Design
from elasticai.creator.vhdl.design_base.ports import Port
from elasticai.creator.vhdl.savable import Path


class Sequential(Design):
    def __init__(
        self,
        sub_designs: list[Design],
        *,
        name: str,
    ) -> None:
        super().__init__(name)
        self._subdesigns = sub_designs
        self._connections: dict[tuple[str, str], tuple[str, str]] = (
            self._build_connections_map()
        )
        self._port = self._build_port()
        self._library_name_for_instances = "work"
        self._architecture_name_for_instances = "rtl"

    def _qualified_signal_name(self, instance: str, signal: str) -> str:
        return f"{instance}_{signal}"

    def _build_port(self) -> Port:
        def get_designs_by_name(name: str) -> list[Design]:
            return [d for d in self._subdesigns if d.name == name]

        def connected_to_self_source(
            sink_source: tuple[tuple[str, str], tuple[str, str]], self_source_name: str
        ) -> bool:
            return sink_source[1] == (self.name, self_source_name)

        def connected_to_self_sink(sink_source, self_sink_name):
            return sink_source[0] == (self.name, self_sink_name)

        def sink_design_name(sink_source):
            return sink_source[0][0]

        def source_design_name(
            sink_source: tuple[tuple[str, str], tuple[str, str]]
        ) -> str:
            return sink_source[1][0]

        def get_connected_designs(get_name, is_connected) -> list[Design]:
            return reduce(
                lambda a, b: a + b,
                map(
                    get_designs_by_name,
                    map(
                        get_name,
                        filter(is_connected, self._connections.items()),
                    ),
                ),
                [],
            )

        width = {k: 1 for k in ("x", "y_address", "x_address", "y")}
        sink_keys = ("y", "x_address")
        source_keys = ("x", "y_address")

        for key in sink_keys:
            connected = partial(connected_to_self_sink, self_sink_name=key)
            for d in get_connected_designs(source_design_name, connected):
                width[key] = d.port[key].width

        for key in source_keys:
            connected = partial(connected_to_self_source, self_source_name=key)
            for d in get_connected_designs(sink_design_name, connected):
                width[key] = d.port[key].width

        return create_port(
            x_width=width["x"],
            y_width=width["y"],
            x_address_width=width["x_address"],
            y_address_width=width["y_address"],
        )

    @property
    def port(self) -> Port:
        return self._port

    def _save_subdesigns(self, destination: Path) -> None:
        for design in self._subdesigns:
            design.save_to(destination.create_subpath(design.name))

    def _instance_names(self) -> list[str]:
        return [f"i_{design.name}" for design in self._subdesigns]

    def _build_connections_map(self) -> dict[tuple[str, str], tuple[str, str]]:
        named_ports = [(d.name, d.port) for d in self._subdesigns]
        nodes = [
            DataFlowNode(
                name=n[0],
                sinks=tuple(s.name for s in n[1].incoming),
                sources=tuple(s.name for s in n[1].outgoing),
            )
            for n in named_ports
        ]
        top = DataFlowNode.top(self.name)
        autowirer = AutoWirer()
        autowirer.wire(top, graph=nodes)
        return autowirer.connections()

    def _generate_connections_code(self) -> list[str]:
        def generate_name(node_name: str, signal_name: str) -> str:
            if node_name == self.name:
                return signal_name
            else:
                return "_".join(("i", node_name, signal_name))

        map = {
            generate_name(*k): generate_name(*v) for k, v in self._connections.items()
        }

        lines = create_connections_using_to_from_pairs(map)
        lines = list(sorted(lines))
        return lines

    def _instance_name_and_design_pairs(self):
        yield from zip(self._instance_names(), self._subdesigns)

    def _generate_instantiations(self) -> list[str]:
        instantiations: list[str] = list()
        for instance, design in self._instance_name_and_design_pairs():
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
                for instance_id, instance in self._instance_name_and_design_pairs()
            )
        )

    @property
    def _x_address_width(self) -> int:
        return self.port["x_address"].width

    @property
    def _y_address_width(self) -> int:
        return self.port["y_address"].width

    @property
    def _x_width(self) -> int:
        return self.port["x"].width

    @property
    def _y_width(self) -> int:
        return self.port["y"].width

    def save_to(self, destination: Path):
        self._save_subdesigns(destination)
        network_template = InProjectTemplate(
            package=module_to_package(self.__module__),
            file_name="network.tpl.vhd",
            parameters=dict(
                layer_connections=self._generate_connections_code(),
                layer_instantiations=self._generate_instantiations(),
                signal_definitions=self._generate_signal_definitions(),
                x_address_width=str(self._x_address_width),
                y_address_width=str(self._y_address_width),
                x_width=str(self._x_width),
                y_width=str(self._y_width),
                layer_name=self.name,
            ),
        )
        destination.create_subpath(self.name).as_file(".vhd").write(network_template)
