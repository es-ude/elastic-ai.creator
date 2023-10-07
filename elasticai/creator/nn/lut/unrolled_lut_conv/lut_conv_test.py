from lut.unrolled_lut_conv.lut_conv import LutConv


def test_depthwise_conv_two_in_channels_kernel_size_six():
    expected = """library ieee;
use ieee.std_logic_1164.all;

entity my_lut_conv is
    port (
        enable : in std_logic;
        clock  : in std_logic;
        x   : in std_logic_vector(12-1 downto 0);
        y  : out std_logic_vector(2-1 downto 0);
    );
end;

architecture rtl of my_lut_conv is
begin
    i_my_lut_conv_lut_0_0 : entity work.my_lut_conv_lut_0
        port map (
            x => x(12-1 downto 6),
            y => y(1),
            enable => enable,
            clock => clock
        );
    i_my_lut_conv_lut_1_0 : entity work.my_lut_conv_lut_1
        port map (
            x => x(6-1 downto 0),
            y => y(0),
            enable => enable,
            clock => clock
        );
end;
"""
    lut0 = L
    conv = LutConv()
