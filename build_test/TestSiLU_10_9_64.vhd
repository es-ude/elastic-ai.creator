library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;               -- for type conversions

entity TestSiLU_10_9_64 is
    port (
        enable : in std_logic;
        clock  : in std_logic;
        x      : in std_logic_vector(10-1 downto 0);
        y      : out std_logic_vector(10-1 downto 0)
    );
end TestSiLU_10_9_64;

architecture rtl of TestSiLU_10_9_64 is
    signal signed_x : signed(10-1 downto 0) := (others=>'0');
    signal signed_y : signed(10-1 downto 0) := (others=>'0');
begin
    signed_x <= signed(x);
    y <= std_logic_vector(signed_y);
    TestSiLU_10_9_64_process : process(x)
    begin
        if signed_x <= -512 then signed_y <= to_signed(-138, 10);
        elsif signed_x <= -496 then signed_y <= to_signed(-137, 10);
        elsif signed_x <= -480 then signed_y <= to_signed(-136, 10);
        elsif signed_x <= -463 then signed_y <= to_signed(-134, 10);
        elsif signed_x <= -447 then signed_y <= to_signed(-133, 10);
        elsif signed_x <= -431 then signed_y <= to_signed(-131, 10);
        elsif signed_x <= -415 then signed_y <= to_signed(-129, 10);
        elsif signed_x <= -398 then signed_y <= to_signed(-127, 10);
        elsif signed_x <= -382 then signed_y <= to_signed(-124, 10);
        elsif signed_x <= -366 then signed_y <= to_signed(-122, 10);
        elsif signed_x <= -350 then signed_y <= to_signed(-119, 10);
        elsif signed_x <= -333 then signed_y <= to_signed(-116, 10);
        elsif signed_x <= -317 then signed_y <= to_signed(-113, 10);
        elsif signed_x <= -301 then signed_y <= to_signed(-109, 10);
        elsif signed_x <= -285 then signed_y <= to_signed(-106, 10);
        elsif signed_x <= -268 then signed_y <= to_signed(-102, 10);
        elsif signed_x <= -252 then signed_y <= to_signed(-98, 10);
        elsif signed_x <= -236 then signed_y <= to_signed(-94, 10);
        elsif signed_x <= -220 then signed_y <= to_signed(-89, 10);
        elsif signed_x <= -203 then signed_y <= to_signed(-84, 10);
        elsif signed_x <= -187 then signed_y <= to_signed(-79, 10);
        elsif signed_x <= -171 then signed_y <= to_signed(-74, 10);
        elsif signed_x <= -155 then signed_y <= to_signed(-69, 10);
        elsif signed_x <= -139 then signed_y <= to_signed(-63, 10);
        elsif signed_x <= -122 then signed_y <= to_signed(-57, 10);
        elsif signed_x <= -106 then signed_y <= to_signed(-51, 10);
        elsif signed_x <= -90 then signed_y <= to_signed(-44, 10);
        elsif signed_x <= -74 then signed_y <= to_signed(-38, 10);
        elsif signed_x <= -57 then signed_y <= to_signed(-31, 10);
        elsif signed_x <= -41 then signed_y <= to_signed(-23, 10);
        elsif signed_x <= -25 then signed_y <= to_signed(-16, 10);
        elsif signed_x <= -9 then signed_y <= to_signed(-8, 10);
        elsif signed_x <= 8 then signed_y <= to_signed(0, 10);
        elsif signed_x <= 24 then signed_y <= to_signed(8, 10);
        elsif signed_x <= 40 then signed_y <= to_signed(16, 10);
        elsif signed_x <= 56 then signed_y <= to_signed(25, 10);
        elsif signed_x <= 73 then signed_y <= to_signed(34, 10);
        elsif signed_x <= 89 then signed_y <= to_signed(44, 10);
        elsif signed_x <= 105 then signed_y <= to_signed(53, 10);
        elsif signed_x <= 121 then signed_y <= to_signed(63, 10);
        elsif signed_x <= 138 then signed_y <= to_signed(73, 10);
        elsif signed_x <= 154 then signed_y <= to_signed(83, 10);
        elsif signed_x <= 170 then signed_y <= to_signed(94, 10);
        elsif signed_x <= 186 then signed_y <= to_signed(104, 10);
        elsif signed_x <= 202 then signed_y <= to_signed(115, 10);
        elsif signed_x <= 219 then signed_y <= to_signed(127, 10);
        elsif signed_x <= 235 then signed_y <= to_signed(138, 10);
        elsif signed_x <= 251 then signed_y <= to_signed(150, 10);
        elsif signed_x <= 267 then signed_y <= to_signed(161, 10);
        elsif signed_x <= 284 then signed_y <= to_signed(174, 10);
        elsif signed_x <= 300 then signed_y <= to_signed(186, 10);
        elsif signed_x <= 316 then signed_y <= to_signed(199, 10);
        elsif signed_x <= 332 then signed_y <= to_signed(211, 10);
        elsif signed_x <= 349 then signed_y <= to_signed(225, 10);
        elsif signed_x <= 365 then signed_y <= to_signed(238, 10);
        elsif signed_x <= 381 then signed_y <= to_signed(251, 10);
        elsif signed_x <= 397 then signed_y <= to_signed(265, 10);
        elsif signed_x <= 414 then signed_y <= to_signed(279, 10);
        elsif signed_x <= 430 then signed_y <= to_signed(293, 10);
        elsif signed_x <= 446 then signed_y <= to_signed(307, 10);
        elsif signed_x <= 462 then signed_y <= to_signed(321, 10);
        elsif signed_x <= 479 then signed_y <= to_signed(337, 10);
        elsif signed_x <= 495 then signed_y <= to_signed(351, 10);
        else signed_y <= to_signed(366, 10);
        end if;
    end process;
end rtl;
