library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;               -- for type conversions

entity TestPReLU_10_9_64 is
    port (
        enable : in std_logic;
        clock  : in std_logic;
        x      : in std_logic_vector(10-1 downto 0);
        y      : out std_logic_vector(10-1 downto 0)
    );
end TestPReLU_10_9_64;

architecture rtl of TestPReLU_10_9_64 is
    signal signed_x : signed(10-1 downto 0) := (others=>'0');
    signal signed_y : signed(10-1 downto 0) := (others=>'0');
begin
    signed_x <= signed(x);
    y <= std_logic_vector(signed_y);
    TestPReLU_10_9_64_process : process(x)
    begin
        if signed_x <= -512 then signed_y <= to_signed(-130, 10);
        elsif signed_x <= -496 then signed_y <= to_signed(-126, 10);
        elsif signed_x <= -480 then signed_y <= to_signed(-122, 10);
        elsif signed_x <= -463 then signed_y <= to_signed(-118, 10);
        elsif signed_x <= -447 then signed_y <= to_signed(-114, 10);
        elsif signed_x <= -431 then signed_y <= to_signed(-110, 10);
        elsif signed_x <= -415 then signed_y <= to_signed(-106, 10);
        elsif signed_x <= -398 then signed_y <= to_signed(-102, 10);
        elsif signed_x <= -382 then signed_y <= to_signed(-98, 10);
        elsif signed_x <= -366 then signed_y <= to_signed(-94, 10);
        elsif signed_x <= -350 then signed_y <= to_signed(-90, 10);
        elsif signed_x <= -333 then signed_y <= to_signed(-85, 10);
        elsif signed_x <= -317 then signed_y <= to_signed(-81, 10);
        elsif signed_x <= -301 then signed_y <= to_signed(-77, 10);
        elsif signed_x <= -285 then signed_y <= to_signed(-73, 10);
        elsif signed_x <= -268 then signed_y <= to_signed(-69, 10);
        elsif signed_x <= -252 then signed_y <= to_signed(-65, 10);
        elsif signed_x <= -236 then signed_y <= to_signed(-61, 10);
        elsif signed_x <= -220 then signed_y <= to_signed(-57, 10);
        elsif signed_x <= -203 then signed_y <= to_signed(-53, 10);
        elsif signed_x <= -187 then signed_y <= to_signed(-49, 10);
        elsif signed_x <= -171 then signed_y <= to_signed(-45, 10);
        elsif signed_x <= -155 then signed_y <= to_signed(-41, 10);
        elsif signed_x <= -139 then signed_y <= to_signed(-37, 10);
        elsif signed_x <= -122 then signed_y <= to_signed(-33, 10);
        elsif signed_x <= -106 then signed_y <= to_signed(-29, 10);
        elsif signed_x <= -90 then signed_y <= to_signed(-25, 10);
        elsif signed_x <= -74 then signed_y <= to_signed(-21, 10);
        elsif signed_x <= -57 then signed_y <= to_signed(-16, 10);
        elsif signed_x <= -41 then signed_y <= to_signed(-12, 10);
        elsif signed_x <= -25 then signed_y <= to_signed(-8, 10);
        elsif signed_x <= -9 then signed_y <= to_signed(-4, 10);
        elsif signed_x <= 8 then signed_y <= to_signed(0, 10);
        elsif signed_x <= 24 then signed_y <= to_signed(16, 10);
        elsif signed_x <= 40 then signed_y <= to_signed(32, 10);
        elsif signed_x <= 56 then signed_y <= to_signed(48, 10);
        elsif signed_x <= 73 then signed_y <= to_signed(65, 10);
        elsif signed_x <= 89 then signed_y <= to_signed(81, 10);
        elsif signed_x <= 105 then signed_y <= to_signed(97, 10);
        elsif signed_x <= 121 then signed_y <= to_signed(113, 10);
        elsif signed_x <= 138 then signed_y <= to_signed(130, 10);
        elsif signed_x <= 154 then signed_y <= to_signed(146, 10);
        elsif signed_x <= 170 then signed_y <= to_signed(162, 10);
        elsif signed_x <= 186 then signed_y <= to_signed(178, 10);
        elsif signed_x <= 202 then signed_y <= to_signed(194, 10);
        elsif signed_x <= 219 then signed_y <= to_signed(211, 10);
        elsif signed_x <= 235 then signed_y <= to_signed(227, 10);
        elsif signed_x <= 251 then signed_y <= to_signed(243, 10);
        elsif signed_x <= 267 then signed_y <= to_signed(259, 10);
        elsif signed_x <= 284 then signed_y <= to_signed(276, 10);
        elsif signed_x <= 300 then signed_y <= to_signed(292, 10);
        elsif signed_x <= 316 then signed_y <= to_signed(308, 10);
        elsif signed_x <= 332 then signed_y <= to_signed(324, 10);
        elsif signed_x <= 349 then signed_y <= to_signed(341, 10);
        elsif signed_x <= 365 then signed_y <= to_signed(357, 10);
        elsif signed_x <= 381 then signed_y <= to_signed(373, 10);
        elsif signed_x <= 397 then signed_y <= to_signed(389, 10);
        elsif signed_x <= 414 then signed_y <= to_signed(406, 10);
        elsif signed_x <= 430 then signed_y <= to_signed(422, 10);
        elsif signed_x <= 446 then signed_y <= to_signed(438, 10);
        elsif signed_x <= 462 then signed_y <= to_signed(454, 10);
        elsif signed_x <= 479 then signed_y <= to_signed(471, 10);
        elsif signed_x <= 495 then signed_y <= to_signed(487, 10);
        else signed_y <= to_signed(503, 10);
        end if;
    end process;
end rtl;
