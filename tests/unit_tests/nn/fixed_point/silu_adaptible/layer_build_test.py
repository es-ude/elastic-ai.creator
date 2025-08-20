from typing import cast

from elasticai.creator.file_generation.in_memory_path import InMemoryFile, InMemoryPath
from elasticai.creator.nn.fixed_point.precomputed.adaptable_silu import AdaptableSiLU


def test_vhdl_code_matches_expected_adapt_silu() -> None:
    expected = """library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity adapt_silu is
    generic (
        BITWIDTH_INPUT : integer := 8;
        BITWIDTH_OUTPUT : integer := 8
    );
    port (
        enable : in std_logic;
        clock  : in std_logic;
        x      : in std_logic_vector(BITWIDTH_INPUT-1 downto 0);
        y      : out std_logic_vector(BITWIDTH_OUTPUT-1 downto 0)
    );
end adapt_silu;

architecture rtl of adapt_silu is
    signal signed_x : signed(BITWIDTH_INPUT-1 downto 0) := (others=>'0');
    signal signed_y : signed(BITWIDTH_OUTPUT-1 downto 0) := (others=>'0');
begin
    signed_x <= signed(x);
    y <= std_logic_vector(signed_y);

    adapt_silu_process : process(signed_x)
    begin
        if enable = '0' then
            signed_y <= to_signed(0, BITWIDTH_OUTPUT);
        else
            if signed_x <= to_signed(-128, BITWIDTH_OUTPUT) then signed_y <= to_signed(-11, BITWIDTH_OUTPUT);
            elsif signed_x <= to_signed(-43, BITWIDTH_OUTPUT) then signed_y <= to_signed(-18, BITWIDTH_OUTPUT);
            elsif signed_x <= to_signed(42, BITWIDTH_OUTPUT) then signed_y <= to_signed(0, BITWIDTH_OUTPUT);
            else signed_y <= to_signed(67, BITWIDTH_OUTPUT);
            end if;
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
