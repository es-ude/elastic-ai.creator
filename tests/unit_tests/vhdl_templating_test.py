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
        "  constant ADDRESS_WIDTH: integer := 16;begin  d_in <= address_in;",
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
