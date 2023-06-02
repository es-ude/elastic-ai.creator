from itertools import chain

from elasticai.creator.hdl.auto_wire_protocols.autowiring import AutoWirer, DataFlowNode
from elasticai.creator.hdl.code_generation.template import (
    InProjectTemplate,
    module_to_package,
)
from elasticai.creator.hdl.design_base.design import Design
from elasticai.creator.hdl.design_base.ports import Port
from elasticai.creator.hdl.design_base.std_signals import (
    clock,
    done,
    enable,
    x,
    x_address,
    y,
    y_address,
)
from elasticai.creator.hdl.savable import Path
from elasticai.creator.hdl.vhdl.code_generation import create_instance
from elasticai.creator.hdl.vhdl.code_generation.code_generation import (
    create_connections_using_to_from_pairs,
    create_signal_definitions,
)


class Sequential(Design):
    def __init__(
        self,
        sub_designs: list[Design],
        *,
        x_width: int,
        y_width: int,
        x_address_width: int,
        y_address_width: int,
        name: str,
    ) -> None:
        super().__init__(name)
        self._x_width = x_width
        self._y_width = y_width
        self._x_address_width = x_address_width
        self._y_address_width = y_address_width
        self._library_name_for_instances = "work"
        self._architecture_name_for_instances = "rtl"
        self._subdesigns = sub_designs

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
        for design in self._subdesigns:
            design.save_to(destination.create_subpath(design.name))

    def _instance_names(self) -> list[str]:
        return [f"i_{design.name}" for design in self._subdesigns]

    def _generate_connections(self) -> list[str]:
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

        def generate_name(node_name: str, signal_name: str) -> str:
            if node_name == self.name:
                return signal_name
            else:
                return "_".join(("i", node_name, signal_name))

        connections = {
            generate_name(*k): generate_name(*v)
            for k, v in autowirer.connections().items()
        }
        sinks = sorted(connections.keys())
        lines = []
        for sink in sinks:
            lines.append(f"{sink} <= {connections[sink]};")

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

    def save_to(self, destination: Path):
        self._save_subdesigns(destination)
        network_template = InProjectTemplate(
            package=module_to_package(self.__module__),
            file_name="network.tpl.vhd",
            parameters=dict(
                layer_connections=self._generate_connections(),
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
