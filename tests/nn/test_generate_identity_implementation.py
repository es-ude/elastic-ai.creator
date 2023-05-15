import re
from typing import cast

import pytest
from pytest_bdd import given, parsers, scenarios, then, when

from elasticai.creator.in_memory_path import InMemoryFile, InMemoryPath
from elasticai.creator.nn.vhdl.identity.layer import BufferedIdentity

scenarios("features/generate_identity_implementation.feature")


@pytest.fixture
def build_root() -> InMemoryPath:
    return InMemoryPath("build", parent=None)


@given(
    parsers.parse(
        "layer with {input_features:d} input features and bit width of {bit_width:d}"
    ),
    target_fixture="identity_layer",
)
def identity_layer(input_features: int, bit_width: int) -> BufferedIdentity:
    return BufferedIdentity(num_input_features=input_features, total_bits=bit_width)


@when("translating and saving hw implementation", target_fixture="generated_code")
def generated_code(
    build_root: InMemoryPath, identity_layer: BufferedIdentity
) -> list[str]:
    savable = identity_layer.translate("fpidentity")
    savable.save_to(build_root)
    identity_file = cast(InMemoryFile, build_root["fpidentity"])
    return identity_file.text


def _extract_vector_width(code: list[str], signal: str) -> int:
    pattern = rf"{signal} .* std_logic_vector\(([0-9]*)-1 downto 0\)"
    for line in code:
        found_match = re.search(pattern, line)
        if found_match is not None:
            return int(found_match.group(1))
    raise ValueError(f"Given code does not contain the signal {signal}")


@then(parsers.parse("width of signal {signal1} and {signal2} are equal to {width:d}"))
def signal_widths_are_equal(
    generated_code: list[str], signal1: str, signal2: str, width: int
) -> None:
    signal1_width = _extract_vector_width(generated_code, signal1)
    signal2_width = _extract_vector_width(generated_code, signal2)
    assert signal1_width == signal2_width == width
