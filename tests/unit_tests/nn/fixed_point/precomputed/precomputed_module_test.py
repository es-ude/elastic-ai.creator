from typing import cast

import torch

from elasticai.creator.file_generation.in_memory_path import InMemoryFile, InMemoryPath
from elasticai.creator.nn.fixed_point.precomputed.precomputed_module import (
    PrecomputedModule,
)


def test_vhdl_code_matches_expected_for_precomputed_module() -> None:
    expected = """library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;               -- for type conversions

entity precomputed is
    port (
        enable : in std_logic;
        clock  : in std_logic;
        x      : in std_logic_vector(8-1 downto 0);
        y      : out std_logic_vector(8-1 downto 0)
    );
end precomputed;

architecture rtl of precomputed is
    signal signed_x : signed(8-1 downto 0) := (others=>'0');
    signal signed_y : signed(8-1 downto 0) := (others=>'0');
begin
    signed_x <= signed(x);
    y <= std_logic_vector(signed_y);
    precomputed_process : process(x)
    begin
        if signed_x <= to_signed(-128, 8) then signed_y <= to_signed(-4, 8);
        elsif signed_x <= to_signed(-43, 8) then signed_y <= to_signed(-4, 8);
        elsif signed_x <= to_signed(42, 8) then signed_y <= to_signed(0, 8);
        else signed_y <= to_signed(4, 8);
        end if;
    end process;
end rtl;
""".splitlines()
    tanh = PrecomputedModule(
        base_module=torch.nn.Tanh(),
        total_bits=8,
        frac_bits=2,
        num_steps=4,
        sampling_intervall=(-float("inf"), float("inf")),
    )
    build_path = InMemoryPath("build", parent=None)
    design = tanh.create_design("precomputed")
    design.save_to(build_path)
    actual = cast(InMemoryFile, build_path["precomputed"]).text
    for text in actual:
        print(text)
    assert actual == expected
