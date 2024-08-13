from typing import cast

import pytest

from elasticai.creator.file_generation.in_memory_path import InMemoryFile, InMemoryPath

from .design import Conv1dDesign


@pytest.fixture
def conv1d_design() -> Conv1dDesign:
    return Conv1dDesign(
        name="conv1d",
        total_bits=16,
        frac_bits=8,
        in_channels=1,
        out_channels=2,
        kernel_size=3,
        signal_length=4,
        weights=[[[1, 1, 1]], [[1, 1, 1]]],
        bias=[1, 1],
    )


def save_design(design: Conv1dDesign) -> dict[str, str]:
    destination = InMemoryPath("conv1d", parent=None)
    design.save_to(destination)
    files = cast(list[InMemoryFile], list(destination.children.values()))
    return {file.name: "\n".join(file.text) for file in files}


def test_saved_design_contains_needed_files(conv1d_design: Conv1dDesign) -> None:
    saved_files = save_design(conv1d_design)

    expected_files = {"conv1d_w_rom.vhd", "conv1d_b_rom.vhd", "conv1d.vhd", "conv1d_fxp_MAC_RoundToZero.vhd", "fxp_mac.vhd"}
    actual_files = set(saved_files.keys())

    assert expected_files == actual_files


def test_weight_rom_code_generated_correctly(conv1d_design: Conv1dDesign) -> None:
    expected_code = """library ieee;
    use ieee.std_logic_1164.all;
    use ieee.std_logic_unsigned.all;
entity conv1d_w_rom is
    port (
        clk : in std_logic;
        en : in std_logic;
        addr : in std_logic_vector(3-1 downto 0);
        data : out std_logic_vector(16-1 downto 0)
    );
end entity conv1d_w_rom;
architecture rtl of conv1d_w_rom is
    type conv1d_w_rom_array_t is array (0 to 2**3-1) of std_logic_vector(16-1 downto 0);
    signal ROM : conv1d_w_rom_array_t:=("0000000000000001","0000000000000001","0000000000000001","0000000000000001","0000000000000001","0000000000000001","0000000000000000","0000000000000000");
    attribute rom_style : string;
    attribute rom_style of ROM : signal is "auto";
begin
    ROM_process: process(clk)
    begin
        if rising_edge(clk) then
            if (en = '1') then
                data <= ROM(conv_integer(addr));
            end if;
        end if;
    end process ROM_process;
end architecture rtl;"""
    saved_files = save_design(conv1d_design)
    actual_code = saved_files["conv1d_w_rom.vhd"]
    assert expected_code == actual_code


def test_bias_rom_code_generated_correctly(conv1d_design: Conv1dDesign) -> None:
    expected_code = """library ieee;
    use ieee.std_logic_1164.all;
    use ieee.std_logic_unsigned.all;
entity conv1d_b_rom is
    port (
        clk : in std_logic;
        en : in std_logic;
        addr : in std_logic_vector(1-1 downto 0);
        data : out std_logic_vector(16-1 downto 0)
    );
end entity conv1d_b_rom;
architecture rtl of conv1d_b_rom is
    type conv1d_b_rom_array_t is array (0 to 2**1-1) of std_logic_vector(16-1 downto 0);
    signal ROM : conv1d_b_rom_array_t:=("0000000000000001","0000000000000001");
    attribute rom_style : string;
    attribute rom_style of ROM : signal is "auto";
begin
    ROM_process: process(clk)
    begin
        if rising_edge(clk) then
            if (en = '1') then
                data <= ROM(conv_integer(addr));
            end if;
        end if;
    end process ROM_process;
end architecture rtl;"""
    saved_files = save_design(conv1d_design)
    actual_code = saved_files["conv1d_b_rom.vhd"]
    assert expected_code == actual_code

