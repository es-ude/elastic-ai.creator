from typing import cast

from elasticai.creator.hdl.code_generation.code_generation import (
    calculate_address_width,
)
from elasticai.creator.hdl.design_base import std_signals
from elasticai.creator.hdl.vhdl.code_generation.code_generation import (
    create_connections,
    create_instance,
    signal_definition,
)
from elasticai.creator.hdl.vhdl.code_generation.template import Template
from elasticai.creator.in_memory_path import InMemoryFile, InMemoryPath
from elasticai.creator.nn.vhdl.identity.layer import FPIdentity
from elasticai.creator.nn.vhdl.sequential import Sequential

"""
Tests:
  - [x] replace fplinear1d with minimal layer that implements identity
  - [x] remove hardsigmoid cases from tests
  - [ ] check that sequential layer generates a unique name for each instantiated subdesign (entity instance)
  - [ ] check that each of the above names corresponds to an existing generated subdesign
  - [ ] test each section (connections, instantiations, etc.) in isolation
  - [ ] add a second layer with buffer
"""


def test_empty_sequential() -> None:
    module = Sequential(tuple())
    template = SequentialTemplate(
        connections=sorted(["y <= x;", "x_address <= y_address;", "done <= enable;"]),
        instantiations=[],
        signal_definitions=[],
        x_width="1",
        y_width="1",
        x_address_width="1",
        y_address_width="1",
        name="sequential",
    )
    expected = template.lines()
    destination = InMemoryPath("sequential", parent=None)
    design = module.translate("sequential")
    design.save_to(destination)
    assert expected == _extract_code(destination, "sequential")


def test_with_single_layer() -> None:
    num_input_features = 6
    total_bits = 16
    template = _prepare_sequential_template_with_identity(
        num_input_features, total_bits
    )
    expected = template.lines()
    module = Sequential((FPIdentity(num_input_features, total_bits),))
    design = module.translate("sequential")
    destination = InMemoryPath("sequential", parent=None)
    design.save_to(destination)
    assert expected == _extract_code(destination, "sequential")


class SequentialTemplate:
    def __init__(
        self,
        connections: list[str],
        instantiations: list[str],
        signal_definitions: list[str],
        x_width: str,
        y_width: str,
        x_address_width: str,
        y_address_width: str,
        name: str,
    ) -> None:
        self._template = Template(
            "network",
            package="elasticai.creator.hdl.vhdl.designs",
        )
        self._template.update_parameters(
            layer_connections=connections,
            layer_instantiations=instantiations,
            signal_definitions=signal_definitions,
            x_width=x_width,
            y_width=y_width,
            x_address_width=x_address_width,
            y_address_width=y_address_width,
            layer_name=name,
        )

    def lines(self) -> list[str]:
        return self._template.lines()


def _prepare_sequential_template_with_identity(
    num_input_features: int, total_bits: int
) -> SequentialTemplate:
    entity = "fpidentity_0"
    instance = f"i_{entity}"
    connections = create_connections(
        {
            f"{instance}_x": "x",
            "y": f"{instance}_y",
            f"{instance}_enable": "enable",
            f"{instance}_clock": "clock",
            "done": f"{instance}_done",
            f"{instance}_y_address": "y_address",
            "x_address": f"{instance}_x_address",
        }
    )

    instantiations = create_instance(
        name=instance,
        entity=entity,
        architecture="rtl",
        library="work",
        signal_mapping={
            s: f"{instance}_{s}"
            for s in ("x", "y", "clock", "enable", "x_address", "y_address", "done")
        },
    )

    signal_definitions = sorted(
        [
            signal_definition(name=f"{instance}_{signal.name}", width=signal.width)
            for signal in (
                std_signals.x(total_bits),
                std_signals.y(total_bits),
                std_signals.clock(),
                std_signals.enable(),
                std_signals.done(),
                std_signals.x_address(calculate_address_width(num_input_features)),
                std_signals.y_address(calculate_address_width(num_input_features)),
            )
        ]
    )

    template = SequentialTemplate(
        connections=connections,
        instantiations=instantiations,
        signal_definitions=signal_definitions,
        x_width=f"{total_bits}",
        y_width=f"{total_bits}",
        x_address_width=str(calculate_address_width(num_input_features)),
        y_address_width=str(calculate_address_width(num_input_features)),
        name="sequential",
    )
    return template


def _extract_code(destination: InMemoryPath, name: str) -> list[str]:
    return cast(InMemoryFile, destination[name]).text
