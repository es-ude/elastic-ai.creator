from elasticai.creator.hdl.vhdl.language import Connection, Connections, Port, SignalDef
from elasticai.creator.hdl.vhdl.parser.parser import parser


def test_parsing_connections():
    expected = Connections(
        [Connection(_from="y", _to="x"), Connection(_from="b", _to="a")]
    )
    code = "x <= y; a <= b;"
    actual = parser.parse(code)
    assert actual == expected


def test_parsing_port():
    v = parser.parse(
        "port ( my_signal : in std_logic; my_other : out std_logic_vector(5 - 1 downto"
        " 0));"
    )

    assert v == Port(
        [
            SignalDef(n, d, w)
            for n, d, w in (("my_signal", "in", 0), ("my_other", "out", 4))
        ]
    )
