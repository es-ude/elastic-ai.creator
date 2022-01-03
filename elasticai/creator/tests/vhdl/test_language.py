from elasticai.creator.vhdl.language import (
    Entity,
    InterfaceVariable,
    DataType,
)
from unittest import TestCase


class EntityTest(TestCase):
    def test_no_name_entity(self):
        e = Entity("")
        expected = ["entity  is", "end entity ;"]
        actual = list(e())
        self.assertEqual(expected, actual)

    def test_entity_name_tanh(self):
        e = Entity("tanh")
        expected = ["entity tanh is", "end entity tanh;"]
        actual = list(e())
        self.assertEqual(expected, actual)

    def test_entity_with_generic(self):
        e = Entity("identifier")

        e.generic_list = ["test"]

        expected = [
            "entity identifier is",
            "\tgeneric (",
            "\t\ttest",
            "\t);",
            "end entity identifier;",
        ]
        actual = list(e())
        self.assertEqual(expected, actual)

    def test_entity_with_port(self):
        e = Entity("identifier")
        e.port_list = ["test"]
        expected = [
            "entity identifier is",
            "\tport (",
            "\t\ttest",
            "\t);",
            "end entity identifier;",
        ]
        actual = list(e())
        self.assertEqual(expected, actual)

    def test_entity_with_two_variables_in_generic(self):
        e = Entity("identifier")
        e.generic_list = ["first", "second"]
        expected = ["\t\tfirst;", "\t\tsecond"]
        actual = list(e())
        actual = actual[2:4]
        self.assertEqual(expected, actual)

    def test_entity_with_interface_variables(self):
        e = Entity("ident")
        e.generic_list.append(
            InterfaceVariable(
                identifier="my_var", variable_type=DataType.INTEGER, value="16"
            )
        )
        expected = ["\t\tmy_var : integer := 16"]
        actual = list(e())
        actual = actual[2:3]
        self.assertEqual(expected, actual)


example = """
entity tanh is
    generic (
        DATA_WIDTH : integer := 16;
        FRAC_WIDTH : integer := 8
    );
    port (
        x : in signed(DATA_WIDTH-1 downto 0);
        y : out signed(DATA_WIDTH-1 downto 0)
    );
end entity tanh;"""
