from typing import cast

from elasticai.creator.file_generation.in_memory_path import InMemoryFile, InMemoryPath
from elasticai.creator.nn.fixed_point.precomputed.silu import SiLU


def test_vhdl_code_matches_expected_silu() -> None:
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
        if signed_x <= to_signed(-128, 8) then signed_y <= to_signed(-1, 8);
        elsif signed_x <= to_signed(-43, 8) then signed_y <= to_signed(-6, 8);
        elsif signed_x <= to_signed(42, 8) then signed_y <= to_signed(0, 8);
        else signed_y <= to_signed(79, 8);
        end if;
    end process;
end rtl;
""".splitlines()
    actfunc = SiLU(
        total_bits=8,
        frac_bits=5,
        num_steps=4,
        sampling_intervall=(-float("inf"), float("inf")),
    )
    build_path = InMemoryPath("build", parent=None)
    design = actfunc.create_design("sigmoid")
    design.save_to(build_path)
    actual = cast(InMemoryFile, build_path["sigmoid"]).text
    assert actual == expected
