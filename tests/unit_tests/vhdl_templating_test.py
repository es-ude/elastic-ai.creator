from collections.abc import Iterable

import pytest

from elasticai.creator.ir2vhdl import EntityTemplateDirector


@pytest.fixture
def dummy_vhdl_prototype() -> Iterable[str]:
    return [
        "library ieee;",
        "use ieee.std.numeric.all;",
        "use ieee.logic_1644.all;",
        "use work.skeleton_pkg.all;",
        "",
        "entity skeleton is",
        "  generic (",
        "    DATA_WIDTH : natural;",
        "    DATA_DEPTH : natural",
        "  );",
        "  port (",
        "    d_in: std_logic_vector(DATA_WIDTH - 1 downto 0);",
        "    address_in: std_logic_vector(16 - 1 downto 0)",
        "  );",
        "end entity;",
        "",
        "begin architecture rtl of skeleton is",
        "  constant ADDRESS_WIDTH: integer := 16;",
        "  constant WEIGHT : std_logic_vector(5 downto 0) := (others => '0');",
        "begin",
        "  d_in <= address_in;",
        "end architecture;",
    ]


class TestVhdlTemplating:
    def test_replace_entity_name(self, dummy_vhdl_prototype) -> None:
        type_handler = (
            EntityTemplateDirector().set_prototype(dummy_vhdl_prototype).build()
        )
        code = type_handler.substitute(
            {
                "entity": "my_first_hw_component",
            }
        ).splitlines()
        assert (
            "entity my_first_hw_component is",
            "begin architecture rtl of my_first_hw_component is",
        ) == (code[5], code[16])

    def test_replace_generic(self, dummy_vhdl_prototype) -> None:
        type_handler = (
            EntityTemplateDirector()
            .add_generic("data_width")
            .set_prototype("""
generic (
  DATA_WIDTH : natural
);
""")
            .build()
        )
        code = type_handler.substitute(dict(data_width=5)).splitlines()
        assert "  DATA_WIDTH : natural := 5" == code[2]

    def test_replace_first_of_two_generics(self) -> None:
        type_handler = (
            EntityTemplateDirector()
            .set_prototype("""
generic (
    DATA_WIDTH : natural;
    DATA_DEPTH : natural
    );
""")
            .add_generic("data_width")
            .build()
        )
        code = type_handler.substitute(dict(data_width=5)).splitlines()
        assert "    DATA_WIDTH : natural := 5;" == code[2]

    def test_can_change_assigned_value(self) -> None:
        type_handler = (
            EntityTemplateDirector()
            .set_prototype("""
generic (
    DATA_WIDTH : natural := 8
    );
""")
            .add_generic("data_width")
            .build()
        )
        code = type_handler.substitute(dict(data_width=5)).splitlines()
        assert "    DATA_WIDTH : natural := 5" == code[2]

    def test_can_change_value_of_positive_type(self) -> None:
        type_handler = (
            EntityTemplateDirector()
            .set_prototype("""
generic (
    DATA_WIDTH : positive := 8
    );
""")
            .add_generic("data_width")
            .build()
        )
        code = type_handler.substitute(dict(data_width=5)).splitlines()
        assert "    DATA_WIDTH : positive := 5" == code[2]

    def test_can_replace_two_parameters(self) -> None:
        type_handler = (
            EntityTemplateDirector()
            .set_prototype("""
generic (
    DATA_WIDTH : natural;
    DATA_DEPTH : natural
    );
""")
            .add_generic("data_width")
            .add_generic("data_depth")
            .build()
        )
        code = type_handler.substitute(dict(data_width=5, data_depth=3)).splitlines()
        assert "    DATA_WIDTH : natural := 5;" == code[2]
        assert "    DATA_DEPTH : natural := 3" == code[3]

    def test_can_replace_constant(self):
        type_handler = (
            EntityTemplateDirector()
            .set_prototype("""
        constant MY_VALUE : integer := 0;
        """)
            .add_value("MY_VALUE")
            .build()
        )
        print(type_handler.template)

        code = type_handler.substitute(dict(MY_VALUE=3))
        assert "constant MY_VALUE : integer := 3" in code

    def setup(self, prototype, param_name):
        return (
            EntityTemplateDirector()
            .set_prototype(prototype)
            .add_value(param_name)
            .build()
        )

    def test_can_replace_logic_vector(self):
        type_handler = self.setup(
            """constant MY_VALUE : std_logic_vector(5 downto 0) := (others => '0');""",
            "MY_VALUE",
        )

        code = type_handler.substitute(dict(MY_VALUE='"010101"'))
        assert 'constant MY_VALUE : std_logic_vector(5 downto 0) := "010101";' == code

    def test_can_replace_logic_vector_defined_with_generics(self):
        type_handler = self.setup(
            "constant VALUE : std_logic_vector(D_WIDTH - 1 downto 0) := (others => '0');",
            "VALUE",
        )
        code = type_handler.substitute(dict(VALUE='"0000"'))
        assert (
            'constant VALUE : std_logic_vector(D_WIDTH - 1 downto 0) := "0000";' == code
        )
