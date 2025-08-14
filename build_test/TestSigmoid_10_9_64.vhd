library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;               -- for type conversions

entity TestSigmoid_10_9_64 is
    port (
        enable : in std_logic;
        clock  : in std_logic;
        x      : in std_logic_vector(10-1 downto 0);
        y      : out std_logic_vector(10-1 downto 0)
    );
end TestSigmoid_10_9_64;

architecture rtl of TestSigmoid_10_9_64 is
    signal signed_x : signed(10-1 downto 0) := (others=>'0');
    signal signed_y : signed(10-1 downto 0) := (others=>'0');
begin
    signed_x <= signed(x);
    y <= std_logic_vector(signed_y);
    TestSigmoid_10_9_64_process : process(x)
    begin
        if signed_x <= -512 then signed_y <= to_signed(136, 10);
        elsif signed_x <= -496 then signed_y <= to_signed(139, 10);
        elsif signed_x <= -480 then signed_y <= to_signed(142, 10);
        elsif signed_x <= -463 then signed_y <= to_signed(146, 10);
        elsif signed_x <= -447 then signed_y <= to_signed(149, 10);
        elsif signed_x <= -431 then signed_y <= to_signed(152, 10);
        elsif signed_x <= -415 then signed_y <= to_signed(156, 10);
        elsif signed_x <= -398 then signed_y <= to_signed(159, 10);
        elsif signed_x <= -382 then signed_y <= to_signed(163, 10);
        elsif signed_x <= -366 then signed_y <= to_signed(166, 10);
        elsif signed_x <= -350 then signed_y <= to_signed(170, 10);
        elsif signed_x <= -333 then signed_y <= to_signed(174, 10);
        elsif signed_x <= -317 then signed_y <= to_signed(177, 10);
        elsif signed_x <= -301 then signed_y <= to_signed(181, 10);
        elsif signed_x <= -285 then signed_y <= to_signed(185, 10);
        elsif signed_x <= -268 then signed_y <= to_signed(189, 10);
        elsif signed_x <= -252 then signed_y <= to_signed(192, 10);
        elsif signed_x <= -236 then signed_y <= to_signed(196, 10);
        elsif signed_x <= -220 then signed_y <= to_signed(200, 10);
        elsif signed_x <= -203 then signed_y <= to_signed(204, 10);
        elsif signed_x <= -187 then signed_y <= to_signed(208, 10);
        elsif signed_x <= -171 then signed_y <= to_signed(212, 10);
        elsif signed_x <= -155 then signed_y <= to_signed(216, 10);
        elsif signed_x <= -139 then signed_y <= to_signed(219, 10);
        elsif signed_x <= -122 then signed_y <= to_signed(224, 10);
        elsif signed_x <= -106 then signed_y <= to_signed(228, 10);
        elsif signed_x <= -90 then signed_y <= to_signed(232, 10);
        elsif signed_x <= -74 then signed_y <= to_signed(235, 10);
        elsif signed_x <= -57 then signed_y <= to_signed(240, 10);
        elsif signed_x <= -41 then signed_y <= to_signed(244, 10);
        elsif signed_x <= -25 then signed_y <= to_signed(248, 10);
        elsif signed_x <= -9 then signed_y <= to_signed(252, 10);
        elsif signed_x <= 8 then signed_y <= to_signed(256, 10);
        elsif signed_x <= 24 then signed_y <= to_signed(260, 10);
        elsif signed_x <= 40 then signed_y <= to_signed(264, 10);
        elsif signed_x <= 56 then signed_y <= to_signed(268, 10);
        elsif signed_x <= 73 then signed_y <= to_signed(272, 10);
        elsif signed_x <= 89 then signed_y <= to_signed(276, 10);
        elsif signed_x <= 105 then signed_y <= to_signed(280, 10);
        elsif signed_x <= 121 then signed_y <= to_signed(284, 10);
        elsif signed_x <= 138 then signed_y <= to_signed(288, 10);
        elsif signed_x <= 154 then signed_y <= to_signed(292, 10);
        elsif signed_x <= 170 then signed_y <= to_signed(296, 10);
        elsif signed_x <= 186 then signed_y <= to_signed(300, 10);
        elsif signed_x <= 202 then signed_y <= to_signed(304, 10);
        elsif signed_x <= 219 then signed_y <= to_signed(308, 10);
        elsif signed_x <= 235 then signed_y <= to_signed(312, 10);
        elsif signed_x <= 251 then signed_y <= to_signed(316, 10);
        elsif signed_x <= 267 then signed_y <= to_signed(319, 10);
        elsif signed_x <= 284 then signed_y <= to_signed(323, 10);
        elsif signed_x <= 300 then signed_y <= to_signed(327, 10);
        elsif signed_x <= 316 then signed_y <= to_signed(331, 10);
        elsif signed_x <= 332 then signed_y <= to_signed(334, 10);
        elsif signed_x <= 349 then signed_y <= to_signed(338, 10);
        elsif signed_x <= 365 then signed_y <= to_signed(342, 10);
        elsif signed_x <= 381 then signed_y <= to_signed(345, 10);
        elsif signed_x <= 397 then signed_y <= to_signed(349, 10);
        elsif signed_x <= 414 then signed_y <= to_signed(352, 10);
        elsif signed_x <= 430 then signed_y <= to_signed(356, 10);
        elsif signed_x <= 446 then signed_y <= to_signed(359, 10);
        elsif signed_x <= 462 then signed_y <= to_signed(363, 10);
        elsif signed_x <= 479 then signed_y <= to_signed(366, 10);
        elsif signed_x <= 495 then signed_y <= to_signed(369, 10);
        else signed_y <= to_signed(372, 10);
        end if;
    end process;
end rtl;
