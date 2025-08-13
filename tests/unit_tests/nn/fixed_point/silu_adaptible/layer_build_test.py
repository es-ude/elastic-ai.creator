from typing import cast

from elasticai.creator.file_generation.in_memory_path import InMemoryFile, InMemoryPath
from elasticai.creator.nn.fixed_point.precomputed.adaptable_silu import AdaptableSiLU


def test_vhdl_code_matches_expected_adapt_silu() -> None:
    expected = """library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;               -- for type conversions

entity adapt_silu is
    port (
        enable : in std_logic;
        clock  : in std_logic;
        x      : in std_logic_vector(8-1 downto 0);
        y      : out std_logic_vector(8-1 downto 0)
    );
end adapt_silu;

architecture rtl of adapt_silu is
    signal signed_x : signed(8-1 downto 0) := (others=>'0');
    signal signed_y : signed(8-1 downto 0) := (others=>'0');
begin
    signed_x <= signed(x);
    y <= std_logic_vector(signed_y);
    adapt_silu_process : process(x)
    begin
        if signed_x <= -128 then signed_y <= to_signed(-11, 8);
        elsif signed_x <= -43 then signed_y <= to_signed(-18, 8);
        elsif signed_x <= 42 then signed_y <= to_signed(0, 8);
        else signed_y <= to_signed(67, 8);
        end if;
    end process;
end rtl;
""".splitlines()
    sigmoid = AdaptableSiLU(
        total_bits=8,
        frac_bits=6,
        num_steps=4,
        sampling_intervall=(-float("inf"), float("inf")),
    )
    build_path = InMemoryPath("build", parent=None)
    design = sigmoid.create_design("adapt_silu")
    design.save_to(build_path)
    actual = cast(InMemoryFile, build_path["adapt_silu"]).text

    for text in actual:
        print(text)
    assert actual == expected
