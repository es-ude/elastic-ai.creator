from elasticai.creator.ir2vhdl import Implementation
from elasticai.creator_plugins.lutron.lutron import lutron


def test_can_generate_vhdl_for_lutron():
    impl = Implementation(
        name="lut_0",
        type="lutron",
        data={
            "input_size": 2,
            "output_size": 2,
            "truth_table": (
                ("00", "11"),
                ("01", "10"),
                ("10", "01"),
                ("11", "11"),
            ),
        },
    )
    _, lines = lutron(impl)
    vhdl = tuple(map(str.strip, lines))
    expected = (
        "library ieee;",
        "use ieee.std_logic_1164.all;",
        "entity lut_0 is",
        "port (",
        "signal d_in : in std_logic_vector(2 - 1 downto 0);",
        "signal d_out : out std_logic_vector(2 - 1 downto 0)",
        ");",
        "end entity;",
        "",
        "architecture rtl of lut_0 is",
        "begin",
        "process (d_in) is",
        "begin",
        "case d_in is",
        'when b"00" => d_out <= b"11";',
        'when b"01" => d_out <= b"10";',
        'when b"10" => d_out <= b"01";',
        'when b"11" => d_out <= b"11";',
        "when others => d_out <= (others => 'X');",
        "end case;",
        "end process;",
        "end architecture;",
    )
    assert vhdl == expected
