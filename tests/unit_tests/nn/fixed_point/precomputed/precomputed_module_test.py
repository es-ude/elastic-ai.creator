from typing import cast

import torch

from elasticai.creator.file_generation.in_memory_path import InMemoryFile, InMemoryPath
from elasticai.creator.nn.fixed_point.precomputed.precomputed_module import (
    PrecomputedModule,
)


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
""".splitlines()
    tanh = PrecomputedModule(
        base_module=torch.nn.Tanh(),
        total_bits=8,
        frac_bits=2,
        num_steps=5,
        sampling_intervall=(-5, 5),
    )
    build_path = InMemoryPath("build", parent=None)
    design = tanh.create_design("tanh")
    design.save_to(build_path)
    actual = cast(InMemoryFile, build_path["tanh"]).text
    assert actual == expected


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
""".splitlines()
    sigmoid = PrecomputedModule(
        base_module=torch.nn.Sigmoid(),
        total_bits=8,
        frac_bits=2,
        num_steps=5,
        sampling_intervall=(-5, 5),
    )
    build_path = InMemoryPath("build", parent=None)
    design = sigmoid.create_design("sigmoid")
    design.save_to(build_path)
    actual = cast(InMemoryFile, build_path["sigmoid"]).text
    assert actual == expected
