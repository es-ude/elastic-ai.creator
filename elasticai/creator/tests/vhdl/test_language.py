from unittest import TestCase

from elasticai.creator.vhdl.generator.precomputed_scalar_function import (
    precomputed_scalar_function_process,
)
from elasticai.creator.vhdl.language import (
    Architecture,
    ContextClause,
    DataType,
    Entity,
    InterfaceSignal,
    InterfaceVariable,
    LibraryClause,
    Mode,
    PortMap,
    Procedure,
    Process,
    UseClause,
)


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
            "generic (",
            "test",
            ");",
            "end entity identifier;",
        ]
        actual = list(e())
        self.assertEqual(expected, actual)

    def test_entity_with_port(self):
        e = Entity("identifier")
        e.port_list = ["test"]
        expected = [
            "entity identifier is",
            "port (",
            "test",
            ");",
            "end entity identifier;",
        ]
        actual = list(e())
        self.assertEqual(expected, actual)

    def test_entity_with_two_variables_in_generic(self):
        e = Entity("identifier")
        e.generic_list = ["first", "second"]
        expected = ["first;", "second"]
        actual = list(e())
        actual = actual[2:4]
        self.assertEqual(expected, actual)

    def test_entity_with_interface_variables(self):
        e = Entity("ident")
        e.generic_list.append(
            InterfaceVariable(
                identifier="my_var", identifier_type=DataType.INTEGER, value="16"
            )
        )
        expected = ["my_var : integer := 16"]
        actual = list(e())
        actual = actual[2:3]
        self.assertEqual(expected, actual)


class LibraryTest(TestCase):
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


class ProcessTest(TestCase):
    # Note: the precomputed scalar function process is already tested, no need for trying more in- and outputs
    def test_process_empty(self):
        process = Process(
            identifier="some_name",
            input_name="some_input",
            lookup_table_generator_function=precomputed_scalar_function_process(
                x_list=[], y_list=[0]
            ),
        )
        expected = [
            "some_name_process: process(some_input)",
            "begin",
            'y <= "0000000000000000";',
            "end process some_name_process;",
        ]
        actual = list(process())
        self.assertEqual(expected, actual)

    def test_process_with_variables(self):
        process = Process(
            identifier="some_name",
            input_name="some_input",
            lookup_table_generator_function=precomputed_scalar_function_process(
                x_list=[], y_list=[0]
            ),
        )
        process.process_declaration_list = ["variable some_variable_name: integer := 0"]
        process.process_statements_list = [
            "some_variable_name := to_integer(some_variable_name)"
        ]
        expected = [
            "some_name_process: process(some_input)",
            "variable some_variable_name: integer := 0;",
            "begin",
            "some_variable_name := to_integer(some_variable_name);",
            'y <= "0000000000000000";',
            "end process some_name_process;",
        ]
        actual = list(process())
        self.assertEqual(expected, actual)


class InterfaceVariableTest(TestCase):
    def test_InterfaceVariable_empty(self):
        interface_variable = InterfaceVariable(
            identifier="some_variable", identifier_type=DataType.INTEGER
        )
        expected = ["some_variable : integer"]
        actual = list(interface_variable())
        self.assertEqual(expected, actual)

    def test_InterfaceVariable_all_parameters(self):
        interface_variable = InterfaceVariable(
            identifier="my_var",
            identifier_type=DataType.SIGNED,
            mode=Mode.IN,
            range="15 downto 0",
            value="16",
        )
        expected = ["my_var : in signed(15 downto 0) := 16"]
        actual = list(interface_variable())
        self.assertEqual(expected, actual)


class InterfaceSignalTest(TestCase):
    def test_InterfaceSignal(self):
        i = InterfaceSignal(
            identifier="y", mode=Mode.OUT, range="x", identifier_type=DataType.SIGNED
        )
        expected = ["signal y : out signed(x)"]
        actual = list(i())
        # actual = actual[2:3]
        self.assertEqual(expected, actual)


class ArchitectureTest(TestCase):
    def test_Architecture_base(self):
        a = Architecture(design_unit="z")
        expected = ["architecture rtl of z is", "begin", "end architecture rtl;"]
        actual = list(a())
        self.assertSequenceEqual(expected, actual)

    def test_Architecture_with_variables(self):
        a = Architecture(design_unit="z")
        a.architecture_declaration_list.append(
            InterfaceVariable(
                identifier="1", range="1", identifier_type=DataType.SIGNED
            )
        )
        expected = [
            "architecture rtl of z is",
            "1 : signed(1);",
            "begin",
            "end architecture rtl;",
        ]
        actual = list(a())
        self.assertSequenceEqual(expected, actual)

    def test_Architecture_with_architecture_part_as_function(self):
        def function():
            yield "some code"

        a = Architecture(design_unit="z")
        a.architecture_statement_part = function
        expected = [
            "architecture rtl of z is",
            "begin",
            "some code",
            "end architecture rtl;",
        ]
        actual = list(a())
        self.assertSequenceEqual(expected, actual)

    def test_Architecture_with_architecture_part_as_process(self):
        def function():
            yield "some code"

        dummy_process = Process(
            identifier="some name",
            lookup_table_generator_function=function(),
            input_name="x",
        )
        a = Architecture(design_unit="z")
        a.architecture_statement_part = dummy_process
        expected = [
            "architecture rtl of z is",
            "begin",
            "some name_process: process(x)",
            "begin",
            "some code",
            "end process some name_process;",
            "end architecture rtl;",
        ]
        actual = list(a())
        self.assertSequenceEqual(expected, actual)

    def test_Architecture_with_assignment(self):
        a = Architecture(design_unit="z")
        a.architecture_assignment_list.append("A <= B")
        expected = [
            "architecture rtl of z is",
            "begin",
            "A <= B;",
            "end architecture rtl;",
        ]
        actual = list(a())
        self.assertSequenceEqual(expected, actual)

    def test_Architecture_with_port_map(self):
        architecture = Architecture(design_unit="lstm_cell")
        port_map = PortMap(map_name="something", component_name="sigmoid")
        port_map.signal_list.append("x => y")
        port_map.signal_list.append("a => b")
        port_map.signal_list.append("c => d")
        architecture.architecture_port_map_list.append(port_map)

        expected = [
            "architecture rtl of lstm_cell is",
            "begin",
            "something: sigmoid",
            "port map (",
            "x => y,",
            "a => b,",
            "c => d",
            ");",
            "end architecture rtl;",
        ]
        actual = list(architecture())
        self.assertSequenceEqual(expected, actual)

    def test_Architecture_with_process_statements_list(self):
        dummy_process = Process(
            identifier="H_OUT",
            input_name="o,c_new",
        )
        dummy_process.process_statements_list.append(
            "h_new <= shift_right((o*c_new), FRAC_WIDTH)(DATA_WIDTH-1 downto 0)"
        )
        a = Architecture(design_unit="z")
        a.architecture_statement_part = dummy_process
        expected = [
            "architecture rtl of z is",
            "begin",
            "H_OUT_process: process(o,c_new)",
            "begin",
            "h_new <= shift_right((o*c_new), FRAC_WIDTH)(DATA_WIDTH-1 downto 0);",
            "end process H_OUT_process;",
            "end architecture rtl;",
        ]
        actual = list(a())
        self.assertSequenceEqual(expected, actual)


class PortMapTest(TestCase):
    def test_PortMap(self):
        port_map = PortMap(map_name="something", component_name="lstm")
        port_map.signal_list.append("x => y")
        expected = ["something: lstm", "port map (", "x => y", ");"]
        actual = list(port_map())
        self.assertSequenceEqual(expected, actual)


class ProcedureTest(TestCase):
    def test_procedure(self):
        procedure = Procedure(identifier="abc")
        expected = ["procedure abc (", "begin", "end abc;"]
        actual = list(procedure())
        self.assertSequenceEqual(expected, actual)

    def test_procedure_with_declaration_and_statement_list(self):
        procedure = Procedure(identifier="abc")
        procedure.declaration_list = [
            "some_variable : in some_name(some_parameter)",
        ]
        procedure.declaration_list_with_is = [
            "signal some_variable_1 : out some_name(some_parameter))"
        ]
        procedure.statement_list = [
            "xyz <= efg",
        ]
        expected = [
            "procedure abc (",
            "some_variable : in some_name(some_parameter);",
            "signal some_variable_1 : out some_name(some_parameter)) is",
            "begin",
            "xyz <= efg;",
            "end abc;",
        ]
        actual = list(procedure())
        self.assertSequenceEqual(expected, actual)
