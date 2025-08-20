from typing import cast

import elasticai.creator.nn.fixed_point as nn_creator
from elasticai.creator.file_generation.in_memory_path import InMemoryFile, InMemoryPath


def test_vhdl_code_matches_expected_prelu() -> None:
    expected = """library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity prelu is
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
end prelu;

architecture rtl of prelu is
    signal signed_x : signed(BITWIDTH_INPUT-1 downto 0) := (others=>'0');
    signal signed_y : signed(BITWIDTH_OUTPUT-1 downto 0) := (others=>'0');
begin
    signed_x <= signed(x);
    y <= std_logic_vector(signed_y);

    prelu_process : process(signed_x)
    begin
        if enable = '0' then
            signed_y <= to_signed(0, BITWIDTH_OUTPUT);
        else
            if signed_x <= to_signed(-128, BITWIDTH_OUTPUT) then signed_y <= to_signed(-43, BITWIDTH_OUTPUT);
            elsif signed_x <= to_signed(-43, BITWIDTH_OUTPUT) then signed_y <= to_signed(-21, BITWIDTH_OUTPUT);
            elsif signed_x <= to_signed(42, BITWIDTH_OUTPUT) then signed_y <= to_signed(0, BITWIDTH_OUTPUT);
            else signed_y <= to_signed(84, BITWIDTH_OUTPUT);
            end if;
        end if;
    end process;
end rtl;
""".splitlines()
    actfunc = nn_creator.PReLU(
        total_bits=8,
        frac_bits=5,
        num_steps=4,
        sampling_intervall=(-float("inf"), float("inf")),
    )
    build_path = InMemoryPath("build", parent=None)
    design = actfunc.create_design("prelu")
    design.save_to(build_path)
    actual = cast(InMemoryFile, build_path["prelu"]).text
    for text in actual:
        print(text)
    assert actual == expected
