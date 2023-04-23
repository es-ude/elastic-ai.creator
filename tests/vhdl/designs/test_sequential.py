import re
from collections.abc import Iterable
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
  - [x] check that sequential layer generates a unique name for each instantiated subdesign (entity instance)
  - [?] check that each of the above names corresponds to an existing generated subdesign
  - [x] test each section (connections, instantiations, etc.) in isolation
  - [?] add a second layer with buffer
"""


def test_empty_sequential() -> None:
    model = Sequential(tuple())
    actual_code = sequential_code_for_model(model)

    template = Template("network", package="elasticai.creator.hdl.vhdl.designs")
    template.update_parameters(
        layer_connections=sorted(
            ["y <= x;", "x_address <= y_address;", "done <= enable;"]
        ),
        layer_instantiations=[],
        signal_definitions=[],
        x_width="1",
        y_width="1",
        x_address_width="1",
        y_address_width="1",
        layer_name="sequential",
    )
    expected_code = template.lines()

    assert expected_code == actual_code


def test_signal_definitions_for_single_layer_model() -> None:
    sequential_code = sequential_code_for_model(single_layer_model())

    signals = extract_signal_definitions(sequential_code)
    target_signals = signal_definitions_for_identity(
        entity="fpidentity_0", num_input_features=6, total_bits=16
    )

    assert len(signals) == len(target_signals)
    assert set(signals) == set(target_signals)


def test_layer_connections_for_single_layer_model() -> None:
    sequential_code = sequential_code_for_model(single_layer_model())

    connections = extract_layer_connections(sequential_code)
    target_connections = create_connections(
        {
            "i_fpidentity_0_x": "x",
            "y": "i_fpidentity_0_y",
            "i_fpidentity_0_enable": "enable",
            "i_fpidentity_0_clock": "clock",
            "done": "i_fpidentity_0_done",
            "i_fpidentity_0_y_address": "y_address",
            "x_address": "i_fpidentity_0_x_address",
        }
    )

    assert len(connections) == len(target_connections)
    assert set(connections) == set(target_connections)


def test_layer_instantiations_for_single_layer_model() -> None:
    remove_indentation = lambda x: list(map(str.strip, x))

    sequential_code = sequential_code_for_model(single_layer_model())
    generated_code = "\n".join(remove_indentation(sequential_code))

    instantiation = identity_layer_instantiation(entity="fpidentity_0")
    target_instantiation = "\n".join(remove_indentation(instantiation))

    assert target_instantiation in generated_code


def test_unique_name_for_each_subdesign() -> None:
    destination = translate_model(two_layer_model())
    subdesign_files = set(destination.children.keys()) - {"sequential"}

    def layer_name(file_name: str) -> str:
        path = cast(InMemoryPath, destination[file_name])
        code = get_code(path[file_name])
        return extract_layer_name(code)

    unique_layer_names = set(layer_name(file_name) for file_name in subdesign_files)
    assert len(unique_layer_names) == 2


def single_layer_model() -> Sequential:
    return Sequential((FPIdentity(num_input_features=6, total_bits=16),))


def two_layer_model() -> Sequential:
    return Sequential(
        (
            FPIdentity(num_input_features=6, total_bits=16),
            FPIdentity(num_input_features=6, total_bits=16),
        )
    )


def get_code(code_file: InMemoryPath | InMemoryFile) -> list[str]:
    return cast(InMemoryFile, code_file).text


def translate_model(model: Sequential) -> InMemoryPath:
    design = model.translate("sequential")
    destination = InMemoryPath("sequential", parent=None)
    design.save_to(destination)
    return destination


def sequential_code_for_model(model) -> list[str]:
    destination = translate_model(model)
    return get_code(destination["sequential"])


def signal_definitions_for_identity(
    entity: str, num_input_features: int, total_bits: int
) -> list[str]:
    return [
        signal_definition(name=f"i_{entity}_{signal.name}", width=signal.width)
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


def identity_layer_instantiation(entity: str) -> list[str]:
    signals = ["clock", "done", "enable", "x", "x_address", "y", "y_address"]

    return create_instance(
        name=f"i_{entity}",
        entity=entity,
        library="work",
        architecture="rtl",
        signal_mapping={signal: f"i_{entity}_{signal}" for signal in signals},
    )


def _find_all_matches(pattern: str, lines: Iterable[str]) -> list[str]:
    return [match for line in lines for match in re.findall(pattern, line)]


def extract_layer_name(code: Iterable[str]) -> str:
    matches = _find_all_matches(pattern=r"entity\s*(\S*)\s*is", lines=code)
    if len(matches) == 0:
        raise ValueError("Code does not contain a layer name.")
    return matches[0]


def extract_signal_definitions(code: list[str]) -> list[str]:
    return _find_all_matches(pattern=r"\s*(signal .*;)", lines=code)


def extract_layer_connections(code: list[str]) -> list[str]:
    return _find_all_matches(pattern=r"\s*(.*<=.*;)", lines=code)
