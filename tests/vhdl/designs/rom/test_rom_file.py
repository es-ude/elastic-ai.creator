import re
from collections.abc import Callable

import pytest

from elasticai.creator.in_memory_path import InMemoryPath
from elasticai.creator.vhdl.designs.rom import Rom

from .utils import extract_rom_values


@pytest.fixture
def rom_name():
    return "test_rom"


@pytest.fixture
def build_root():
    return InMemoryPath("build", parent=None)


@pytest.fixture
def rom_destination(build_root, rom_name):
    return build_root.create_subpath(rom_name)


@pytest.fixture
def built_rom_file(build_root, rom_name, rom_destination):
    class Wrapper:
        @property
        def text(self) -> list[str]:
            return build_root[rom_name].text

    return Wrapper()


def test_generating_00_00(rom_destination, rom_name, built_rom_file):
    rom = Rom(rom_name, data_width=8, values_as_integers=[0, 0])
    rom.save_to(rom_destination)
    actual = extract_rom_values(built_rom_file.text)
    assert actual == ("00", "00")


def test_generate_af_bb(rom_destination, rom_name, built_rom_file):
    rom = Rom(
        rom_name,
        data_width=8,
        values_as_integers=[2**4 * 10 + 15, 2**4 * 11 + 11],
    )
    rom.save_to(rom_destination)
    actual = extract_rom_values(built_rom_file.text)
    assert actual == ("af", "bb")


def test_generate_0000(rom_destination, rom_name, built_rom_file):
    rom = Rom(rom_name, data_width=16, values_as_integers=[0, 0])
    rom.save_to(rom_destination)
    actual = extract_rom_values(built_rom_file.text)
    assert actual == ("0000", "0000")


@pytest.fixture
def get_address_width(built_rom_file) -> Callable[[], int]:
    def extract() -> int:
        text = built_rom_file.text
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


def test_address_width_is_1(rom_destination, rom_name, get_address_width):
    rom = Rom(rom_name, data_width=8, values_as_integers=[0, 0])
    rom.save_to(rom_destination)
    assert get_address_width() == 1


def test_address_width_is_3(rom_destination, rom_name, get_address_width):
    rom = Rom(rom_name, data_width=8, values_as_integers=[0, 0, 0, 0, 0])
    rom.save_to(rom_destination)
    assert get_address_width() == 3


def test_rom_values_are_filled_up_from_3_to_4(
    rom_destination, rom_name, built_rom_file
):
    rom = Rom(rom_name, data_width=8, values_as_integers=[1] * 3)
    rom.save_to(rom_destination)
    actual = extract_rom_values(built_rom_file.text)
    assert actual == ("01", "01", "01", "00")


def test_rom_values_are_filled_up_from_18_to_32(
    rom_destination, rom_name, built_rom_file
):
    rom = Rom(rom_name, data_width=8, values_as_integers=[1] * 18)
    rom.save_to(rom_destination)
    actual = extract_rom_values(built_rom_file.text)
    assert actual == tuple(["01"] * 18 + ["00"] * 14)
