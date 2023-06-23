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
def rom_destination(build_root: InMemoryPath, rom_name: str) -> InMemoryPath:
    return build_root.create_subpath(rom_name)


@pytest.fixture
def get_rom_content(build_root: InMemoryPath, rom_name: str) -> Callable[[], list[str]]:
    def build() -> list[str]:
        return cast(InMemoryFile, build_root[rom_name]).text

    return build


@pytest.fixture
def get_address_width(get_rom_content) -> Callable[[], int]:
    def extract() -> int:
        text = get_rom_content()
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
    rom_destination: InMemoryPath,
    rom_name: str,
    get_rom_content: Callable[[], list[str]],
    values_as_integers: list[int],
    expected_rom_values: tuple[str, ...],
) -> None:
    rom = Rom(rom_name, data_width=8, values_as_integers=values_as_integers)
    rom.save_to(rom_destination)
    actual = extract_rom_values(get_rom_content())
    assert actual == expected_rom_values


@pytest.mark.parametrize(
    ("values_as_integers", "expected_address_width"),
    [
        ([0, 0], 1),
        ([0, 0, 0, 0, 0], 3),
    ],
)
def test_address_width_is_set_correctly(
    rom_destination: InMemoryPath,
    rom_name: str,
    get_address_width: Callable[[], int],
    values_as_integers: list[int],
    expected_address_width: int,
) -> None:
    rom = Rom(rom_name, data_width=8, values_as_integers=values_as_integers)
    rom.save_to(rom_destination)
    assert get_address_width() == expected_address_width


@pytest.mark.parametrize(
    ("values_as_integers", "expected_rom_values"),
    [
        ([1] * 3, ("1", "1", "1", "0")),
        ([1] * 18, tuple(["1"] * 18 + ["0"] * 14)),
    ],
)
def test_rom_values_are_filled_up_correctly(
    rom_destination: InMemoryPath,
    rom_name: str,
    get_rom_content: Callable[[], list[str]],
    values_as_integers: list[int],
    expected_rom_values: list[str],
) -> None:
    rom = Rom(rom_name, data_width=8, values_as_integers=values_as_integers)
    rom.save_to(rom_destination)
    actual = extract_rom_values(get_rom_content())
    assert actual == expected_rom_values
