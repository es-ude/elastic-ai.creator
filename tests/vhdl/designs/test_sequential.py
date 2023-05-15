import re
from collections.abc import Iterable
from typing import cast

import pytest

from elasticai.creator.hdl.code_generation.code_generation import (
    calculate_address_width,
)
from elasticai.creator.hdl.code_generation.template import (
    InProjectTemplate,
    TemplateExpander,
)
from elasticai.creator.hdl.design_base import std_signals
from elasticai.creator.hdl.vhdl.code_generation.code_generation import (
    create_connections_using_to_from_pairs,
    create_instance,
    signal_definition,
)
from elasticai.creator.in_memory_path import InMemoryFile, InMemoryPath
from elasticai.creator.nn.vhdl.identity.layer import BaseIdentity, BufferedIdentity
from elasticai.creator.nn.vhdl.sequential import Sequential


def single_layer_model() -> Sequential:
    return Sequential((BufferedIdentity(num_input_features=6, total_bits=16),))


def two_layer_model() -> Sequential:
    return Sequential(
        (
            BufferedIdentity(num_input_features=6, total_bits=16),
            BufferedIdentity(num_input_features=6, total_bits=16),
        )
    )


class TestSequential:
    def test_empty_sequential(self) -> None:
        model = Sequential(tuple())
        actual_code = sequential_code_for_model(model)

        template = InProjectTemplate(
            package="elasticai.creator.hdl.vhdl.designs",
            file_name="network.tpl.vhd",
            parameters=dict(
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
            ),
        )
        expected_code = TemplateExpander(template).lines()
        assert actual_code == expected_code

    def test_unique_name_for_each_subdesign(self) -> None:
        destination = translate_model(two_layer_model())
        subdesign_files = set(destination.children.keys()) - {"sequential"}

        def layer_name(file_name: str) -> str:
            path = cast(InMemoryPath, destination[file_name])
            code = get_code(path[file_name])
            return extract_layer_name(code)

        unique_layer_names = set(layer_name(file_name) for file_name in subdesign_files)
        assert len(unique_layer_names) == 2

    @pytest.mark.parametrize(
        "model,entity_id",
        [(single_layer_model(), 0), (two_layer_model(), 0), (two_layer_model(), 1)],
    )
    def test_signal_definitions(self, model: Sequential, entity_id: int) -> None:
        sequential_code = sequential_code_for_model(model)

        signals = extract_signal_definitions(sequential_code)
        target_signals = signal_definitions_for_identity(
            entity=f"bufferedidentity_{entity_id}", num_input_features=6, total_bits=16
        )

        assert set(target_signals) <= set(signals)

    @pytest.mark.parametrize(
        "model,entity_id",
        [(single_layer_model(), 0), (two_layer_model(), 0), (two_layer_model(), 1)],
    )
    def test_layer_instantiations(self, model: Sequential, entity_id: int) -> None:
        sequential_code = sequential_code_for_model(model)
        generated_code = "\n".join(remove_indentation(sequential_code))

        instantiation = identity_layer_instantiation(
            entity=f"bufferedidentity_{entity_id}"
        )
        target_instantiation = "\n".join(remove_indentation(instantiation))

        assert target_instantiation in generated_code

    def test_layer_connections_for_single_layer_model(self) -> None:
        sequential_code = sequential_code_for_model(single_layer_model())

        connections = extract_layer_connections(sequential_code)
        name = "bufferedidentity"
        target_connections = create_connections_using_to_from_pairs(
            {
                f"i_{name}_0_x": "x",
                "y": f"i_{name}_0_y",
                f"i_{name}_0_enable": "enable",
                f"i_{name}_0_clock": "clock",
                "done": f"i_{name}_0_done",
                f"i_{name}_0_y_address": "y_address",
                "x_address": f"i_{name}_0_x_address",
            }
        )

        assert set(connections) == set(target_connections)

    def test_layer_connections_for_two_layer_model(self) -> None:
        sequential_code = sequential_code_for_model(two_layer_model())
        layer_0 = "i_bufferedidentity_0"
        layer_1 = "i_bufferedidentity_1"
        connections = extract_layer_connections(sequential_code)
        target_connections = create_connections_using_to_from_pairs(
            {
                f"{layer_0}_clock": "clock",
                f"{layer_0}_enable": "enable",
                f"{layer_0}_x": "x",
                "x_address": f"{layer_0}_x_address",
                f"{layer_0}_y_address": f"{layer_1}_x_address",
                f"{layer_1}_x": f"{layer_0}_y",
                f"{layer_1}_enable": f"{layer_0}_done",
                f"{layer_1}_clock": "clock",
                f"{layer_1}_y_address": "y_address",
                "y": f"{layer_1}_y",
                "done": f"{layer_1}_done",
            }
        )

        assert set(connections) == set(target_connections)


def get_code(code_file: InMemoryPath | InMemoryFile) -> list[str]:
    return cast(InMemoryFile, code_file).text


def translate_model(model: Sequential) -> InMemoryPath:
    design = model.translate("sequential")
    destination = InMemoryPath("sequential", parent=None)
    design.save_to(destination)
    return destination


def sequential_code_for_model(model: Sequential) -> list[str]:
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


def remove_indentation(code: Iterable[str]) -> list[str]:
    return list(map(str.strip, code))


def _find_all_matches(pattern: str, lines: Iterable[str]) -> list[str]:
    return [match for line in lines for match in re.findall(pattern, line)]


def extract_layer_name(code: Iterable[str]) -> str:
    matches = _find_all_matches(pattern=r"entity\s*(\S*)\s*is", lines=code)
    if len(matches) == 0:
        raise ValueError("Code does not contain a layer name.")
    return matches[0]


def extract_signal_definitions(code: Iterable[str]) -> list[str]:
    return _find_all_matches(pattern=r"\s*(signal .*;)", lines=code)


def extract_layer_connections(code: Iterable[str]) -> list[str]:
    return _find_all_matches(pattern=r"\s*(.*<=.*;)", lines=code)
