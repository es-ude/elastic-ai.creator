from itertools import chain

from elasticai.creator.hdl.code_generation.template import module_to_package
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
from elasticai.creator.hdl.savable import Path
from elasticai.creator.hdl.vhdl.code_generation import create_instance
from elasticai.creator.hdl.vhdl.code_generation.code_generation import (
    create_connections_using_to_from_pairs,
    create_signal_definitions,
)
from elasticai.creator.hdl.vhdl.code_generation.template import InProjectTemplate


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
        prefixes = [f"{name}_" for name in self._instance_names()]
        connections = {f"{name}clock": "clock" for name in prefixes}

        def wire_pair(a, b):
            connections.update(
                {
                    f"{a}y_address": f"{b}x_address",
                    f"{b}x": f"{a}y",
                    f"{b}enable": f"{a}done",
                }
            )

        for a, b in zip(prefixes[:-1], prefixes[1:]):
            wire_pair(a, b)

        def wire_top_to_start_and_end(start, end):
            connections.update(
                {
                    "done": f"{end}done",
                    "y": f"{end}y",
                    f"{end}y_address": f"y_address",
                    f"{start}x": "x",
                    f"{start}enable": "enable",
                    f"x_address": f"{start}x_address",
                }
            )

        if len(prefixes) == 0:
            connections.update({"x_address": "y_address", "y": "x", "done": "enable"})
        else:
            wire_top_to_start_and_end(prefixes[0], prefixes[-1])

        return create_connections_using_to_from_pairs(connections)

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
        network_implementation = InProjectTemplate(
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
            layer_name=self.name,
        )
        target_file = destination.create_subpath(self.name).as_file(".vhd")
        target_file.write_text(network_implementation.lines())
