from elasticai.creator_plugins.combinatorial.language import Port, VHDLEntity


def test_can_generate_entity():
    expected = [
        "library ieee;",
        "use ieee.std_logic_1164.all;",
        "entity my_entity is",
        "generic (",
        "A : int;",
        "B : positive",
        ");",
        "port (",
        "signal clk : in std_logic;",
        "signal y : out std_logic_vector(3 - 1 downto 0)",
        ");",
        "end entity;",
    ]

    impl = VHDLEntity(
        name="my_entity",
        port=Port(
            inputs=dict(
                clk="std_logic",
            ),
            outputs=dict(y="std_logic_vector(3 - 1 downto 0)"),
        ),
        generics=dict(A="int", B="positive"),
    )
    actual = list(map(str.strip, impl.generate_entity()))
    assert actual == expected
