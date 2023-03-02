from typing import Iterator

from elasticai.creator.hdl.vhdl.code_generation import (
    create_instance,
    signal_definition,
)
from elasticai.creator.hdl.vhdl.code_generation.code_generation import (
    calculate_address_width,
    create_connections,
)
from elasticai.creator.hdl.vhdl.saveable import Path
from elasticai.creator.hdl.vhdl.template import Template

from .design import Design, Port
from .std_signals import clock, done, enable, x, x_address, y, y_address


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

    def _generate_connections(self) -> list[str]:
        last_y = "x"
        last_x_address = "y_address"
        last_done = "enable"
        connections: dict[str, str] = {}
        for instance in self.instances:
            connections[self._qualified_signal_name(instance, "x")] = last_y
            connections[self._qualified_signal_name(instance, "clock")] = "clock"
            connections[self._qualified_signal_name(instance, "enable")] = last_done
            last_y = self._qualified_signal_name(instance, "y")
        connections["y"] = last_y
        connections["x_address"] = last_x_address
        connections["done"] = last_done
        return create_connections(connections)

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
            [
                signal_definition(
                    name=self._qualified_signal_name(
                        instance=instance, signal=signal.name
                    ),
                    width=signal.width,
                )
                for instance, signal in [
                    (instance, signal)
                    for instance in self.instances
                    for signal in self.instances[instance].port.signals
                ]
            ]
        )

    def save_to(self, destination: Path):
        network_implementation = Template("network")
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
