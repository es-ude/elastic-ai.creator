import re
from collections.abc import Callable
from typing import cast

import pytest

from elasticai.creator.file_generation.in_memory_path import InMemoryFile, InMemoryPath
from elasticai.creator.vhdl.shared_designs.rom.design import Rom


def extract_rom_values(text: str | list[str]) -> tuple[str, ...]:
    if not isinstance(text, list):
        text = [text]
    values: tuple[str, ...] = tuple()
    for line in text:
        match = re.match(
            r'.*\("([a-f0-9]+(",\s?"[a-f0-9]+)*)"\)',
            line,
        )
        if match is not None:
            array = match.group(1)
            values = tuple(re.split(r'(?:",\s?")', array))

    return values


class TestExtractRomValues:
    @pytest.mark.parametrize(
        ("values_to_extract", "values"),
        [
            ('"0"', ("0",)),
            ('"0", "1"', ("0", "1")),
            ('"10101111", "10111011"', ("10101111", "10111011")),
            ('"10101111","10111011"', ("10101111", "10111011")),
        ],
    )
    def test_extracted_rom_values(
        self, values_to_extract: str, values: tuple[str]
    ) -> None:
        test_line = f"signal ROM : some_name_array_t:=({values_to_extract})"
        assert extract_rom_values(test_line) == values

    def test_can_handle_iteration_over_multiple_lines(self):
        test_lines = [
            "some text",
            'some text that is comparable to array ("0", "1")',
            'some_other_name_array_t:=("10101111", "10111011")',
        ]
        assert extract_rom_values(test_lines) == ("10101111", "10111011")


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
    destination = build_root

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
            match = re.match(r"\s+ROM_ADDR_WIDTH : integer :=\s+(\d+);", line)
            if match:
                actual = int(match.group(1))
                break
        return actual

    return extract


class TestRomDesign:
    @pytest.mark.parametrize(
        ("values_as_integers", "expected_rom_values"),
        [
            ([0, 0], ("00000000", "00000000")),
            ([-81, -69], ("10101111", "10111011")),
        ],
    )
    def test_generating_correct_rom_values(
        self,
        rom_code: Callable[[list[int]], list[str]],
        values_as_integers: list[int],
        expected_rom_values: tuple[str, ...],
    ) -> None:
        code = rom_code(values_as_integers)
        assert extract_rom_values(code) == expected_rom_values

    @pytest.mark.parametrize(
        ("values_as_integers", "expected_rom_values"),
        [
            ([1] * 3, ("00000001", "00000001", "00000001", "00000000")),
            ([1] * 18, tuple(["00000001"] * 18 + ["00000000"] * 14)),
        ],
    )
    def test_rom_values_are_padded_correctly(
        self,
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
        self,
        address_width: Callable[[list[int]], int],
        values_as_integers: list[int],
        expected_address_width: int,
    ) -> None:
        assert address_width(values_as_integers) == expected_address_width
