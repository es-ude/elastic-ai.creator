from typing import cast

from elasticai.creator.in_memory_path import InMemoryFile, InMemoryPath
from elasticai.creator.nn.vhdl.tanh import FPTanh


def test_tanh_output_with_two_steps() -> None:
    expected = """library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;               -- for type conversions

entity tanh is
    port (
        x   : in std_logic_vector(8-1 downto 0);
        y  : out std_logic_vector(8-1 downto 0);
    );
end tanh;

architecture rtl of tanh is
    signal signed_x, signed_y : signed(8-1 downto 0) := (others=>'0');
begin
    signed_x <- signed(x);
    y <- std_logic_vector(signed_y);
    x_addr <- y_addr;
    done <- enable;
    tanh_process : process(x)
    begin
        if signed_x <= 20 then signed_y <= to_signed(3, 8);
        if signed_x <= 0 then signed_y <= to_signed(0, 8);
        else signed_y <= to_signed(-3, 8);
        end if;
    end process;
end rtl;
""".splitlines()
    tanh = FPTanh(total_bits=8, frac_bits=2, num_steps=3, sampling_intervall=(-5, 5))
    build_path = InMemoryPath("build", parent=None)
    design = tanh.translate("tanh")
    design.save_to(build_path)
    actual = cast(InMemoryFile, build_path["tanh"]).text
    assert actual == expected
