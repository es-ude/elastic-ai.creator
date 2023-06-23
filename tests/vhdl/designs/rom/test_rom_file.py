import re
from collections.abc import Callable
from typing import cast

import pytest

from elasticai.creator.in_memory_path import InMemoryFile, InMemoryPath
from elasticai.creator.vhdl.designs.rom import Rom

from .utils import extract_rom_values


@pytest.fixture
def rom_name() -> str:
    return "test_rom"


@pytest.fixture
def build_root() -> InMemoryPath:
    return InMemoryPath("build", parent=None)


@pytest.fixture
def rom_code(
    rom_name: str, build_root: InMemoryPath
) -> Callable[[list[int]], list[str]]:
    destination = build_root.create_subpath(rom_name)

    def generate_code(values_as_integers: list[int]) -> list[str]:
        rom = Rom(name=rom_name, data_width=8, values_as_integers=values_as_integers)
        rom.save_to(destination)
        return cast(InMemoryFile, build_root[rom_name]).text

    return generate_code


@pytest.fixture
def address_width(rom_code) -> Callable[[list[int]], int]:
    def extract(values_as_integers: list[int]) -> int:
        text = rom_code(values_as_integers)
        actual = 0
        for line in text:
            match = re.match(
                r"\s+addr\s?:\s?in std_logic_vector\((\d+)-1 downto 0\);", line
            )
            if match:
                actual = int(match.group(1))
                break
        return actual

    return extract


@pytest.mark.parametrize(
    ("values_as_integers", "expected_rom_values"),
    [
        ([0, 0], ("0", "0")),
        ([175, 187], ("10101111", "10111011")),
    ],
)
def test_generating_correct_rom_values(
    rom_code: Callable[[list[int]], list[str]],
    values_as_integers: list[int],
    expected_rom_values: tuple[str, ...],
) -> None:
    code = rom_code(values_as_integers)
    assert extract_rom_values(code) == expected_rom_values


@pytest.mark.parametrize(
    ("values_as_integers", "expected_rom_values"),
    [
        ([1] * 3, ("1", "1", "1", "0")),
        ([1] * 18, tuple(["1"] * 18 + ["0"] * 14)),
    ],
)
def test_rom_values_are_padded_correctly(
    rom_code: Callable[[list[int]], list[str]],
    values_as_integers: list[int],
    expected_rom_values: list[str],
) -> None:
    code = rom_code(values_as_integers)
    assert extract_rom_values(code) == expected_rom_values


@pytest.mark.parametrize(
    ("values_as_integers", "expected_address_width"),
    [
        ([0, 0], 1),
        ([0, 0, 0, 0, 0], 3),
    ],
)
def test_address_width_is_set_correctly(
    address_width: Callable[[list[int]], int],
    values_as_integers: list[int],
    expected_address_width: int,
) -> None:
    assert address_width(values_as_integers) == expected_address_width
