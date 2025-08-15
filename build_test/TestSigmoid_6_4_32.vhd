library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;               -- for type conversions

entity TestSigmoid_6_4_32 is
    port (
        enable : in std_logic;
        clock  : in std_logic;
        x      : in std_logic_vector(6-1 downto 0);
        y      : out std_logic_vector(6-1 downto 0)
    );
end TestSigmoid_6_4_32;

architecture rtl of TestSigmoid_6_4_32 is
    signal signed_x : signed(6-1 downto 0) := (others=>'0');
    signal signed_y : signed(6-1 downto 0) := (others=>'0');
begin
    signed_x <= signed(x);
    y <= std_logic_vector(signed_y);
    TestSigmoid_6_4_32_process : process(x)
    begin
        if signed_x <= -32 then signed_y <= to_signed(2, 6);
        elsif signed_x <= -30 then signed_y <= to_signed(2, 6);
        elsif signed_x <= -28 then signed_y <= to_signed(2, 6);
        elsif signed_x <= -26 then signed_y <= to_signed(2, 6);
        elsif signed_x <= -24 then signed_y <= to_signed(3, 6);
        elsif signed_x <= -22 then signed_y <= to_signed(3, 6);
        elsif signed_x <= -20 then signed_y <= to_signed(3, 6);
        elsif signed_x <= -18 then signed_y <= to_signed(4, 6);
        elsif signed_x <= -16 then signed_y <= to_signed(4, 6);
        elsif signed_x <= -14 then signed_y <= to_signed(4, 6);
        elsif signed_x <= -12 then signed_y <= to_signed(5, 6);
        elsif signed_x <= -10 then signed_y <= to_signed(5, 6);
        elsif signed_x <= -8 then signed_y <= to_signed(6, 6);
        elsif signed_x <= -6 then signed_y <= to_signed(6, 6);
        elsif signed_x <= -4 then signed_y <= to_signed(7, 6);
        elsif signed_x <= -2 then signed_y <= to_signed(7, 6);
        elsif signed_x <= 1 then signed_y <= to_signed(8, 6);
        elsif signed_x <= 3 then signed_y <= to_signed(8, 6);
        elsif signed_x <= 5 then signed_y <= to_signed(9, 6);
        elsif signed_x <= 7 then signed_y <= to_signed(9, 6);
        elsif signed_x <= 9 then signed_y <= to_signed(10, 6);
        elsif signed_x <= 11 then signed_y <= to_signed(10, 6);
        elsif signed_x <= 13 then signed_y <= to_signed(11, 6);
        elsif signed_x <= 15 then signed_y <= to_signed(11, 6);
        elsif signed_x <= 17 then signed_y <= to_signed(12, 6);
        elsif signed_x <= 19 then signed_y <= to_signed(12, 6);
        elsif signed_x <= 21 then signed_y <= to_signed(12, 6);
        elsif signed_x <= 23 then signed_y <= to_signed(13, 6);
        elsif signed_x <= 25 then signed_y <= to_signed(13, 6);
        elsif signed_x <= 27 then signed_y <= to_signed(13, 6);
        elsif signed_x <= 29 then signed_y <= to_signed(14, 6);
        else signed_y <= to_signed(14, 6);
        end if;
    end process;
end rtl;
