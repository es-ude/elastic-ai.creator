from collections.abc import Iterable

import pytest

from elasticai.creator.ir2vhdl import VhdlEntityIr, process_vhdl_template


@pytest.fixture
def dummy_vhdl_template() -> Iterable[str]:
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
        "  constant ADDRESS_WIDTH: integer := 16;" "begin" "  d_in <= address_in;",
        "end architecture;",
    ]


class TestVhdlTemplating:
    @pytest.mark.skip
    def test_replace_entity_name(self, dummy_vhdl_template) -> None:
        type_handler = process_vhdl_template(dummy_vhdl_template)
        entity = VhdlEntityIr(name="my_first_hw_component", type="skeleton_v1")
        code = list(type_handler(entity))
        assert (
            "entity my_first_hw_component is",
            "begin architecture rtl of my_first_hw_component is",
        ) == (code[5], code[16])
