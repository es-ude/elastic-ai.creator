library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;               -- for type conversions

entity TestSiLUAdaptible_8_4_32 is
    port (
        enable : in std_logic;
        clock  : in std_logic;
        x      : in std_logic_vector(8-1 downto 0);
        y      : out std_logic_vector(8-1 downto 0)
    );
end TestSiLUAdaptible_8_4_32;

architecture rtl of TestSiLUAdaptible_8_4_32 is
    signal signed_x : signed(8-1 downto 0) := (others=>'0');
    signal signed_y : signed(8-1 downto 0) := (others=>'0');
begin
    signed_x <= signed(x);
    y <= std_logic_vector(signed_y);
    TestSiLUAdaptible_8_4_32_process : process(x)
    begin
        if signed_x <= -128 then signed_y <= to_signed(0, 8);
        elsif signed_x <= -120 then signed_y <= to_signed(0, 8);
        elsif signed_x <= -112 then signed_y <= to_signed(0, 8);
        elsif signed_x <= -103 then signed_y <= to_signed(0, 8);
        elsif signed_x <= -95 then signed_y <= to_signed(0, 8);
        elsif signed_x <= -87 then signed_y <= to_signed(0, 8);
        elsif signed_x <= -79 then signed_y <= to_signed(0, 8);
        elsif signed_x <= -70 then signed_y <= to_signed(-1, 8);
        elsif signed_x <= -62 then signed_y <= to_signed(-1, 8);
        elsif signed_x <= -54 then signed_y <= to_signed(-1, 8);
        elsif signed_x <= -46 then signed_y <= to_signed(-2, 8);
        elsif signed_x <= -38 then signed_y <= to_signed(-3, 8);
        elsif signed_x <= -29 then signed_y <= to_signed(-4, 8);
        elsif signed_x <= -21 then signed_y <= to_signed(-4, 8);
        elsif signed_x <= -13 then signed_y <= to_signed(-4, 8);
        elsif signed_x <= -5 then signed_y <= to_signed(-3, 8);
        elsif signed_x <= 4 then signed_y <= to_signed(0, 8);
        elsif signed_x <= 12 then signed_y <= to_signed(5, 8);
        elsif signed_x <= 20 then signed_y <= to_signed(11, 8);
        elsif signed_x <= 28 then signed_y <= to_signed(19, 8);
        elsif signed_x <= 37 then signed_y <= to_signed(29, 8);
        elsif signed_x <= 45 then signed_y <= to_signed(38, 8);
        elsif signed_x <= 53 then signed_y <= to_signed(47, 8);
        elsif signed_x <= 61 then signed_y <= to_signed(55, 8);
        elsif signed_x <= 69 then signed_y <= to_signed(64, 8);
        elsif signed_x <= 78 then signed_y <= to_signed(73, 8);
        elsif signed_x <= 86 then signed_y <= to_signed(81, 8);
        elsif signed_x <= 94 then signed_y <= to_signed(89, 8);
        elsif signed_x <= 102 then signed_y <= to_signed(98, 8);
        elsif signed_x <= 111 then signed_y <= to_signed(107, 8);
        elsif signed_x <= 119 then signed_y <= to_signed(115, 8);
        else signed_y <= to_signed(123, 8);
        end if;
    end process;
end rtl;
