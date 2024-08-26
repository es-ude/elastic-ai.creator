import pytest

from elasticai.creator.nn.identity.layer import BufferedIdentity
from tests.design_file_structure import design_file_structure

from .layer import Sequential


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
end rtl;
"""
        assert expected == get_code(Sequential())

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
    signal i_bufferedidentity_0_x_address : std_logic_vector({address_width-1} downto 0) := (others => '0');
    signal i_bufferedidentity_0_y : std_logic_vector(3 downto 0) := (others => '0');
    signal i_bufferedidentity_0_y_address : std_logic_vector({address_width-1} downto 0) := (others => '0');
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
end rtl;
"""
        model = Sequential(
            BufferedIdentity(num_input_features=number_of_layer_inputs, total_bits=4)
        )
        assert expected == get_code(model)


def get_code(model: Sequential) -> str:
    design = model.create_design("sequential")
    return design_file_structure(design)["sequential.vhd"]
