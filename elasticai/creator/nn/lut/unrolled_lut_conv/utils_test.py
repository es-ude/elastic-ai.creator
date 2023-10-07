from lut.unrolled_lut_conv.utils import instantiate_lut


def test_instantiate_6_to_1_lut():
    actual = instantiate_lut(
        lut_name="conv_lut_0", instance_index=0, input_slice=(0, 6), output_slice=(2, 3)
    )
    expected = """i_conv_lut_0_0 : entity work.conv_lut_0(rtl)
port map(
    clock => clock,
    enable => enable,
    x => x(6-1 downto 0),
    y => y(3-1 downto 2)
);"""
    assert actual == expected


def test_instantiate_second_4_to_2_lut():
    actual = instantiate_lut(
        lut_name="conv_lut_0", instance_index=1, input_slice=(4, 8), output_slice=(2, 4)
    )
    expected = """i_conv_lut_0_1 : entity work.conv_lut_0(rtl)
port map(
    clock => clock,
    enable => enable,
    x => x(8-1 downto 4),
    y => y(4-1 downto 2)
);"""
    assert actual == expected
