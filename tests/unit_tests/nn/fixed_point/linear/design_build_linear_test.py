from typing import cast

import pytest

from elasticai.creator.file_generation.in_memory_path import InMemoryFile, InMemoryPath
from elasticai.creator.nn.fixed_point.linear.design import LinearDesign


@pytest.fixture
def linear_design() -> LinearDesign:
    return LinearDesign(
        name="linear",
        in_feature_num=3,
        out_feature_num=2,
        total_bits=16,
        frac_bits=8,
        weights=[[1, 3, 5], [2, 4, 6]],
        bias=[7, 8],
    )


def save_design(design: LinearDesign) -> dict[str, str]:
    destination = InMemoryPath("linear", parent=None)
    design.save_to(destination)
    files = cast(list[InMemoryFile], list(destination.children.values()))
    return {file.name: "\n".join(file.text) for file in files}


def test_saved_design_contains_needed_files(linear_design: LinearDesign) -> None:
    saved_files = save_design(linear_design)
    expected_files = {"linear_rom.vhd", "linear.vhd"}
    actual_files = set(saved_files.keys())
    assert expected_files == actual_files


def test_linear_rom_code_generated_correctly(linear_design: LinearDesign) -> None:
    expected_code = """library ieee;
use ieee.numeric_std.all;
use ieee.std_logic_1164.all;

entity linear_rom is
    generic (
        ROM_ADDR_WIDTH : integer := 3;
        ROM_DATA_WIDTH : integer := 16
    );
    port (
        clk : in std_logic;
        en : in std_logic;
        addr : in std_logic_vector(ROM_ADDR_WIDTH-1 downto 0);
        data : out std_logic_vector(ROM_DATA_WIDTH-1 downto 0)
    );
end entity linear_rom;
architecture rtl of linear_rom is
    type linear_rom_array_t is array (0 to 2**ROM_ADDR_WIDTH-1) of std_logic_vector(ROM_DATA_WIDTH-1 downto 0);
    signal ROM : linear_rom_array_t:=("0000000000000001", "0000000000000011", "0000000000000101", "0000000000000111", "0000000000000010", "0000000000000100", "0000000000000110", "0000000000001000");
    attribute rom_style : string;
    attribute rom_style of ROM : signal is "auto";
begin
    ROM_process: process(addr)
    begin
        if (en = '1') then
            data <= ROM(to_integer(unsigned(addr)));
        end if;
    end process ROM_process;
end architecture rtl;"""
    saved_files = save_design(linear_design)
    actual_code = saved_files["linear_rom.vhd"]
    print(actual_code)
    assert expected_code == actual_code
