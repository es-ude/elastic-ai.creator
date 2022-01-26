from elasticai.creator.vhdl.language import (
    Entity,
    InterfaceVariable,
    DataType,
    ContextClause,
    LibraryClause,
    UseClause,
    Process,
)
import unittest
from unittest import TestCase
from elasticai.creator.vhdl.generator.generator_functions import (
    precomputed_scalar_function_process,
)


class LanguageTest(TestCase):
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

    # FIXME: do not implement test for the architecture because it is already tested in the other pull request

    def test_library_with_extra_libraries(self):
        lib = ContextClause(
            library_clause=LibraryClause(logical_name_list=["ieee", "work"]),
            use_clause=UseClause(
                selected_names=[
                    "ieee.std_logic_1164.all",
                    "ieee.numeric_std.all",
                    "work.all",
                ]
            ),
        )
        expected = [
            "library ieee, work;",
            "use ieee.std_logic_1164.all;",
            "use ieee.numeric_std.all;",
            "use work.all;",
        ]
        actual = list(lib())
        self.assertEqual(expected, actual)

    # Note: the precomputed scalar function process is already tested, no need for trying more in- and outputs
    def test_process_empty(self):
        process = Process(
            identifier="some_name",
            input="some_input",
            lookup_table_generator_function=precomputed_scalar_function_process(
                x_list=[], y_list=[0]
            ),
        )
        expected = [
            "some_name_process: process(some_input)",
            "begin",
            '\ty <= "0000000000000000";\n\t\t',
            "end process some_name_process;",
        ]
        actual = list(process())
        self.assertEqual(expected, actual)

    def test_process_with_variables(self):
        process = Process(
            identifier="some_name",
            input="some_input",
            lookup_table_generator_function=precomputed_scalar_function_process(
                x_list=[], y_list=[0]
            ),
        )
        process.item_declaration_list = ["variable some_variable_name: integer := 0"]
        process.sequential_statements_list = [
            "some_variable_name := to_integer(some_variable_name)"
        ]
        expected = [
            "some_name_process: process(some_input)",
            "\tvariable some_variable_name: integer := 0;",
            "begin",
            "\tsome_variable_name := to_integer(some_variable_name);",
            '\ty <= "0000000000000000";\n\t\t',
            "end process some_name_process;",
        ]
        actual = list(process())
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
