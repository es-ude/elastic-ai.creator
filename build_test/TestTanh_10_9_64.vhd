library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;               -- for type conversions

entity TestTanh_10_9_64 is
    port (
        enable : in std_logic;
        clock  : in std_logic;
        x      : in std_logic_vector(10-1 downto 0);
        y      : out std_logic_vector(10-1 downto 0)
    );
end TestTanh_10_9_64;

architecture rtl of TestTanh_10_9_64 is
    signal signed_x : signed(10-1 downto 0) := (others=>'0');
    signal signed_y : signed(10-1 downto 0) := (others=>'0');
begin
    signed_x <= signed(x);
    y <= std_logic_vector(signed_y);
    TestTanh_10_9_64_process : process(x)
    begin
        if signed_x <= -512 then signed_y <= to_signed(-393, 10);
        elsif signed_x <= -496 then signed_y <= to_signed(-387, 10);
        elsif signed_x <= -480 then signed_y <= to_signed(-380, 10);
        elsif signed_x <= -463 then signed_y <= to_signed(-372, 10);
        elsif signed_x <= -447 then signed_y <= to_signed(-364, 10);
        elsif signed_x <= -431 then signed_y <= to_signed(-356, 10);
        elsif signed_x <= -415 then signed_y <= to_signed(-347, 10);
        elsif signed_x <= -398 then signed_y <= to_signed(-338, 10);
        elsif signed_x <= -382 then signed_y <= to_signed(-329, 10);
        elsif signed_x <= -366 then signed_y <= to_signed(-319, 10);
        elsif signed_x <= -350 then signed_y <= to_signed(-309, 10);
        elsif signed_x <= -333 then signed_y <= to_signed(-298, 10);
        elsif signed_x <= -317 then signed_y <= to_signed(-288, 10);
        elsif signed_x <= -301 then signed_y <= to_signed(-276, 10);
        elsif signed_x <= -285 then signed_y <= to_signed(-265, 10);
        elsif signed_x <= -268 then signed_y <= to_signed(-252, 10);
        elsif signed_x <= -252 then signed_y <= to_signed(-240, 10);
        elsif signed_x <= -236 then signed_y <= to_signed(-227, 10);
        elsif signed_x <= -220 then signed_y <= to_signed(-214, 10);
        elsif signed_x <= -203 then signed_y <= to_signed(-200, 10);
        elsif signed_x <= -187 then signed_y <= to_signed(-186, 10);
        elsif signed_x <= -171 then signed_y <= to_signed(-172, 10);
        elsif signed_x <= -155 then signed_y <= to_signed(-158, 10);
        elsif signed_x <= -139 then signed_y <= to_signed(-143, 10);
        elsif signed_x <= -122 then signed_y <= to_signed(-128, 10);
        elsif signed_x <= -106 then signed_y <= to_signed(-112, 10);
        elsif signed_x <= -90 then signed_y <= to_signed(-97, 10);
        elsif signed_x <= -74 then signed_y <= to_signed(-82, 10);
        elsif signed_x <= -57 then signed_y <= to_signed(-65, 10);
        elsif signed_x <= -41 then signed_y <= to_signed(-49, 10);
        elsif signed_x <= -25 then signed_y <= to_signed(-33, 10);
        elsif signed_x <= -9 then signed_y <= to_signed(-17, 10);
        elsif signed_x <= 8 then signed_y <= to_signed(0, 10);
        elsif signed_x <= 24 then signed_y <= to_signed(16, 10);
        elsif signed_x <= 40 then signed_y <= to_signed(32, 10);
        elsif signed_x <= 56 then signed_y <= to_signed(48, 10);
        elsif signed_x <= 73 then signed_y <= to_signed(64, 10);
        elsif signed_x <= 89 then signed_y <= to_signed(80, 10);
        elsif signed_x <= 105 then signed_y <= to_signed(96, 10);
        elsif signed_x <= 121 then signed_y <= to_signed(111, 10);
        elsif signed_x <= 138 then signed_y <= to_signed(127, 10);
        elsif signed_x <= 154 then signed_y <= to_signed(142, 10);
        elsif signed_x <= 170 then signed_y <= to_signed(157, 10);
        elsif signed_x <= 186 then signed_y <= to_signed(171, 10);
        elsif signed_x <= 202 then signed_y <= to_signed(185, 10);
        elsif signed_x <= 219 then signed_y <= to_signed(200, 10);
        elsif signed_x <= 235 then signed_y <= to_signed(213, 10);
        elsif signed_x <= 251 then signed_y <= to_signed(226, 10);
        elsif signed_x <= 267 then signed_y <= to_signed(239, 10);
        elsif signed_x <= 284 then signed_y <= to_signed(252, 10);
        elsif signed_x <= 300 then signed_y <= to_signed(264, 10);
        elsif signed_x <= 316 then signed_y <= to_signed(275, 10);
        elsif signed_x <= 332 then signed_y <= to_signed(287, 10);
        elsif signed_x <= 349 then signed_y <= to_signed(298, 10);
        elsif signed_x <= 365 then signed_y <= to_signed(308, 10);
        elsif signed_x <= 381 then signed_y <= to_signed(318, 10);
        elsif signed_x <= 397 then signed_y <= to_signed(328, 10);
        elsif signed_x <= 414 then signed_y <= to_signed(338, 10);
        elsif signed_x <= 430 then signed_y <= to_signed(347, 10);
        elsif signed_x <= 446 then signed_y <= to_signed(355, 10);
        elsif signed_x <= 462 then signed_y <= to_signed(363, 10);
        elsif signed_x <= 479 then signed_y <= to_signed(372, 10);
        elsif signed_x <= 495 then signed_y <= to_signed(379, 10);
        else signed_y <= to_signed(386, 10);
        end if;
    end process;
end rtl;
