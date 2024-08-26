import torch

from tests.temporary_file_structure import get_savable_file_structure

from .precomputed_module import PrecomputedModule


def test_vhdl_code_matches_expected_for_tanh_as_base_module() -> None:
    expected = """library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;               -- for type conversions

entity tanh is
    port (
        enable : in std_logic;
        clock  : in std_logic;
        x      : in std_logic_vector(8-1 downto 0);
        y      : out std_logic_vector(8-1 downto 0)
    );
end tanh;

architecture rtl of tanh is
    signal signed_x : signed(8-1 downto 0) := (others=>'0');
    signal signed_y : signed(8-1 downto 0) := (others=>'0');
begin
    signed_x <= signed(x);
    y <= std_logic_vector(signed_y);
    tanh_process : process(x)
    begin
        if signed_x <= -20 then signed_y <= to_signed(-3, 8);
        elsif signed_x <= -10 then signed_y <= to_signed(-3, 8);
        elsif signed_x <= 0 then signed_y <= to_signed(0, 8);
        elsif signed_x <= 10 then signed_y <= to_signed(3, 8);
        else signed_y <= to_signed(3, 8);
        end if;
    end process;
end rtl;
"""
    tanh = PrecomputedModule(
        base_module=torch.nn.Tanh(),
        total_bits=8,
        frac_bits=2,
        num_steps=5,
        sampling_intervall=(-5, 5),
    )
    design = tanh.create_design("tanh")
    files = get_savable_file_structure(design)
    assert expected == files["tanh.vhd"]


def test_vhdl_code_matches_expected_for_sigmoid_as_base_module() -> None:
    expected = """library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;               -- for type conversions

entity sigmoid is
    port (
        enable : in std_logic;
        clock  : in std_logic;
        x      : in std_logic_vector(8-1 downto 0);
        y      : out std_logic_vector(8-1 downto 0)
    );
end sigmoid;

architecture rtl of sigmoid is
    signal signed_x : signed(8-1 downto 0) := (others=>'0');
    signal signed_y : signed(8-1 downto 0) := (others=>'0');
begin
    signed_x <= signed(x);
    y <= std_logic_vector(signed_y);
    sigmoid_process : process(x)
    begin
        if signed_x <= -20 then signed_y <= to_signed(0, 8);
        elsif signed_x <= -10 then signed_y <= to_signed(0, 8);
        elsif signed_x <= 0 then signed_y <= to_signed(2, 8);
        elsif signed_x <= 10 then signed_y <= to_signed(3, 8);
        else signed_y <= to_signed(3, 8);
        end if;
    end process;
end rtl;
"""
    sigmoid = PrecomputedModule(
        base_module=torch.nn.Sigmoid(),
        total_bits=8,
        frac_bits=2,
        num_steps=5,
        sampling_intervall=(-5, 5),
    )
    design = sigmoid.create_design("sigmoid")
    files = get_savable_file_structure(design)
    assert expected == files["sigmoid.vhd"]
