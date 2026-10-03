from typing import cast

from elasticai.creator.file_generation.in_memory_path import InMemoryFile, InMemoryPath
from elasticai.creator.nn.fixed_point.precomputed.tanh import Tanh


def test_vhdl_code_matches_expected_tanh() -> None:
    expected = """library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity sigmoid is
    generic (
        BITWIDTH_INPUT : integer := 8;
        BITWIDTH_OUTPUT : integer := 8;
        PIPELINE_ENABLE : boolean := false
    );
    port (
        enable : in std_logic;
        clock  : in std_logic;
        x      : in std_logic_vector(BITWIDTH_INPUT-1 downto 0);
        y      : out std_logic_vector(BITWIDTH_OUTPUT-1 downto 0)
    );
end sigmoid;

architecture rtl of sigmoid is
    signal signed_x : signed(BITWIDTH_INPUT-1 downto 0) := (others=>'0');
    signal signed_y : signed(BITWIDTH_OUTPUT-1 downto 0) := (others=>'0');

    function compute(sx : signed; en : std_logic) return signed is
    begin
        if en = '0' then
            return to_signed(0, BITWIDTH_OUTPUT);
        elsif sx <= to_signed(-43, BITWIDTH_INPUT) then return to_signed(-32, BITWIDTH_OUTPUT);
        elsif sx <= to_signed(42, BITWIDTH_INPUT) then return to_signed(0, BITWIDTH_OUTPUT);
        else return to_signed(32, BITWIDTH_OUTPUT);
        end if;
    end function;
begin
    signed_x <= signed(x);
    y        <= std_logic_vector(signed_y);

    comb_gen : if not PIPELINE_ENABLE generate
        precomputed_process : process(signed_x, enable)
        begin
            signed_y <= compute(signed_x, enable);
        end process;
    end generate;

    pipe_gen : if PIPELINE_ENABLE generate
        precomputed_process : process(clock)
        begin
            if rising_edge(clock) then
                signed_y <= compute(signed_x, enable);
            end if;
        end process;
    end generate;
end rtl;
""".splitlines()
    actfunc = Tanh(
        total_bits=8,
        frac_bits=5,
        num_steps=4,
        sampling_intervall=(-float("inf"), float("inf")),
    )
    build_path = InMemoryPath("build", parent=None)
    design = actfunc.create_design("sigmoid")
    design.save_to(build_path)
    actual = cast(InMemoryFile, build_path["sigmoid"]).text
    for text in actual:
        print(text)
    assert actual == expected
