from typing import cast

import pytest

from elasticai.creator.file_generation.in_memory_path import InMemoryFile, InMemoryPath
from elasticai.creator.nn.identity.layer import BufferedIdentity
from elasticai.creator.nn.sequential.layer import Sequential


class TestSequential:
    def test_empty_sequential_passes_through_signals(self) -> None:
        expected = """library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

library work;
use work.all;

entity sequential is
    port (
        enable: in std_logic;
        clock: in std_logic;

        x_address: out std_logic_vector(1-1 downto 0);
        y_address: in std_logic_vector(1-1 downto 0);

        x: in std_logic_vector(1-1 downto 0);
        y: out std_logic_vector(1-1 downto 0);

        done: out std_logic
    );
end sequential;

architecture rtl of sequential is
begin
    done <= enable;
    x_address <= y_address;
    y <= x;
    --------------------------------------------------------------------------------
    -- Instantiate all layers
    --------------------------------------------------------------------------------
end rtl;"""
        actual_code = "\n".join(sequential_layer_code_for_model(Sequential()))
        assert actual_code == expected

    @pytest.mark.parametrize(
        "number_of_layer_inputs, address_width", [(2, 1), (3, 2), (8, 3)]
    )
    def test_single_layer_model(self, number_of_layer_inputs, address_width) -> None:
        expected = f"""library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

library work;
use work.all;

entity sequential is
    port (
        enable: in std_logic;
        clock: in std_logic;

        x_address: out std_logic_vector({address_width}-1 downto 0);
        y_address: in std_logic_vector({address_width}-1 downto 0);

        x: in std_logic_vector(4-1 downto 0);
        y: out std_logic_vector(4-1 downto 0);

        done: out std_logic
    );
end sequential;

architecture rtl of sequential is
    signal i_bufferedidentity_0_clock : std_logic := '0';
    signal i_bufferedidentity_0_done : std_logic := '0';
    signal i_bufferedidentity_0_enable : std_logic := '0';
    signal i_bufferedidentity_0_x : std_logic_vector(3 downto 0) := (others => '0');
    signal i_bufferedidentity_0_x_address : std_logic_vector({address_width - 1} downto 0) := (others => '0');
    signal i_bufferedidentity_0_y : std_logic_vector(3 downto 0) := (others => '0');
    signal i_bufferedidentity_0_y_address : std_logic_vector({address_width - 1} downto 0) := (others => '0');
begin
    done <= i_bufferedidentity_0_done;
    i_bufferedidentity_0_clock <= clock;
    i_bufferedidentity_0_enable <= enable;
    i_bufferedidentity_0_x <= x;
    i_bufferedidentity_0_y_address <= y_address;
    x_address <= i_bufferedidentity_0_x_address;
    y <= i_bufferedidentity_0_y;
    --------------------------------------------------------------------------------
    -- Instantiate all layers
    --------------------------------------------------------------------------------
    i_bufferedidentity_0 : entity work.bufferedidentity_0(rtl)
    port map(
      clock => i_bufferedidentity_0_clock,
      done => i_bufferedidentity_0_done,
      enable => i_bufferedidentity_0_enable,
      x => i_bufferedidentity_0_x,
      x_address => i_bufferedidentity_0_x_address,
      y => i_bufferedidentity_0_y,
      y_address => i_bufferedidentity_0_y_address
    );
end rtl;"""
        model = Sequential(
            BufferedIdentity(num_input_features=number_of_layer_inputs, total_bits=4)
        )
        actual_code = "\n".join(sequential_layer_code_for_model(model))
        assert actual_code == expected


def get_code(code_file: InMemoryPath | InMemoryFile) -> list[str]:
    return cast(InMemoryFile, code_file).text


def translate_model(model: Sequential) -> InMemoryPath:
    design = model.create_design("sequential")
    destination = InMemoryPath("sequential", parent=None)
    design.save_to(destination)
    return destination


def sequential_layer_code_for_model(model: Sequential) -> list[str]:
    destination = translate_model(model)
    return get_code(destination["sequential"])
