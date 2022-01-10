import unittest
from elasticai.creator.vhdl.generator.general_strings import (
    get_libraries_string,
    get_architecture_header_string,
    get_architecture_end_string,
    get_entity_or_component_string,
    get_signal_definitions_string,
    get_variable_definitions_string,
)


class GeneralStringsTest(unittest.TestCase):
    def test_generate_libraries(self) -> None:
        expected_import_lines = [
            "library ieee;",
            "use ieee.std_logic_1164.all;",
            "use ieee.numeric_std.all;               -- for type conversions",
        ]
        lib_string = get_libraries_string()
        for i in range(0, 3):
            self.assertEqual(expected_import_lines[i], lib_string.splitlines()[i])

    def test_generate_entity_with_generic_without_vector_length_width(self) -> None:
        expected_entity_lines = [
            "entity sigmoid_tb is",
            "\tgeneric (",
            "\t\tDATA_WIDTH : integer := 16;",
            "\t\tFRAC_WIDTH : integer := 8",
            "\t);",
            "\tport (",
            "\t\tclk : out std_logic",
            "\t);",
            "end entity sigmoid_tb;",
        ]
        entity_string = get_entity_or_component_string(
            entity_or_component="entity",
            entity_or_component_name="sigmoid_tb",
            data_width=16,
            frac_width=8,
            variables_dict={"clk": "out std_logic"},
        )
        for i in range(len(expected_entity_lines)):
            self.assertEqual(expected_entity_lines[i], entity_string.splitlines()[i])

    def test_generate_entity_with_generic(self) -> None:
        expected_entity_lines = [
            "entity lstm_common_gate_tb is",
            "\tgeneric (",
            "\t\tDATA_WIDTH : integer := 16;",
            "\t\tFRAC_WIDTH : integer := 8;",
            "\t\tVECTOR_LEN_WIDTH : integer := 4",
            "\t);",
            "\tport (",
            "\t\tclk : out std_logic",
            "\t);",
            "end entity lstm_common_gate_tb;",
        ]
        entity_string = get_entity_or_component_string(
            entity_or_component="entity",
            entity_or_component_name="lstm_common_gate_tb",
            data_width=16,
            frac_width=8,
            variables_dict={"clk": "out std_logic"},
            vector_len_width=4,
        )
        for i in range(len(expected_entity_lines)):
            self.assertEqual(expected_entity_lines[i], entity_string.splitlines()[i])

    def test_generate_component(self) -> None:
        expected_component_lines = [
            "component sigmoid is",
            "\tgeneric (",
            "\t\tDATA_WIDTH : integer := 16;",
            "\t\tFRAC_WIDTH : integer := 8",
            "\t);",
            "\tport (",
            "\t\tx : in signed(DATA_WIDTH-1 downto 0);",
            "\t\ty : out signed(DATA_WIDTH-1 downto 0)",
            "\t);",
            "end component sigmoid;",
        ]
        component_string = get_entity_or_component_string(
            entity_or_component="component",
            entity_or_component_name="sigmoid",
            data_width=16,
            frac_width=8,
            variables_dict={
                "x": "in signed(DATA_WIDTH-1 downto 0)",
                "y": "out signed(DATA_WIDTH-1 downto 0)",
            },
            indent="",
        )
        self.assertEqual(expected_component_lines, component_string.splitlines())

    def test_generate_component_with_different_in_and_out_variables(self) -> None:
        expected_component_lines = [
            "component lstm_common_gate is",
            "\tgeneric (",
            "\t\tDATA_WIDTH : integer := 16;",
            "\t\tFRAC_WIDTH : integer := 8;",
            "\t\tVECTOR_LEN_WIDTH : integer := 4",
            "\t);",
            "\tport (",
            "\t\treset : in std_logic;",
            "\t\tclk : in std_logic;",
            "\t\tx : in signed(DATA_WIDTH-1 downto 0);",
            "\t\tw : in signed(DATA_WIDTH-1 downto 0);",
            "\t\tb : in signed(DATA_WIDTH-1 downto 0);",
            "\t\tvector_len : in unsigned(VECTOR_LEN_WIDTH-1 downto 0);",
            "\t\tidx : out unsigned(VECTOR_LEN_WIDTH-1 downto 0);",
            "\t\tready : out std_logic;",
            "\t\ty : out signed(DATA_WIDTH-1 downto 0)",
            "\t);",
            "end component lstm_common_gate;",
        ]
        component_string = get_entity_or_component_string(
            entity_or_component="component",
            entity_or_component_name="lstm_common_gate",
            data_width=16,
            frac_width=8,
            variables_dict={
                "reset": "in std_logic",
                "clk": "in std_logic",
                "x": "in signed(DATA_WIDTH-1 downto 0)",
                "w": "in signed(DATA_WIDTH-1 downto 0)",
                "b": "in signed(DATA_WIDTH-1 downto 0)",
                "vector_len": "in unsigned(VECTOR_LEN_WIDTH-1 downto 0)",
                "idx": "out unsigned(VECTOR_LEN_WIDTH-1 downto 0)",
                "ready": "out std_logic",
                "y": "out signed(DATA_WIDTH-1 downto 0)",
            },
            vector_len_width=4,
            indent="\t",
        )
        for i in range(len(expected_component_lines)):
            self.assertEqual(
                expected_component_lines[i], component_string.splitlines()[i]
            )

    def test_generate_signal_definitions(self) -> None:
        expected_inputs_lines = [
            "    signal clk_period : time := 1 ns;",
            "    signal test_input : signed(16-1 downto 0):=(others=>'0');",
            "    signal test_output : signed(16-1 downto 0);",
        ]
        signal_definition_string = get_signal_definitions_string(
            signal_dict={
                "clk_period": "time := 1 ns",
                "test_input": "signed(16-1 downto 0):=(others=>'0')",
                "test_output": "signed(16-1 downto 0)",
            }
        )
        for i in range(len(expected_inputs_lines)):
            self.assertEqual(
                expected_inputs_lines[i], signal_definition_string.splitlines()[i]
            )

    def test_generate_signal_definitions_multiple(self) -> None:
        expected_inputs_lines = [
            "    signal clk_period : time := 2 ps;",
            "    signal clock : std_logic;",
            "    signal reset, ready : std_logic:='0';",
            "    signal X_MEM : RAM_ARRAY :=(others=>(others=>'0'));",
            "    signal W_MEM : RAM_ARRAY:=(others=>(others=>'0'));",
            "    signal x, w, y, b : signed(DATA_WIDTH-1 downto 0):=(others=>'0');",
            "    signal vector_len : unsigned(VECTOR_LEN_WIDTH-1 downto 0):=(others=>'0');",
            "    signal idx : unsigned(VECTOR_LEN_WIDTH-1 downto 0):=(others=>'0');",
        ]
        signal_definition_string = get_signal_definitions_string(
            signal_dict={
                "clk_period": "time := 2 ps",
                "clock": "std_logic",
                "reset, ready": "std_logic:='0'",
                "X_MEM": "RAM_ARRAY :=(others=>(others=>'0'))",
                "W_MEM": "RAM_ARRAY:=(others=>(others=>'0'))",
                "x, w, y, b": "signed(DATA_WIDTH-1 downto 0):=(others=>'0')",
                "vector_len": "unsigned(VECTOR_LEN_WIDTH-1 downto 0):=(others=>'0')",
                "idx": "unsigned(VECTOR_LEN_WIDTH-1 downto 0):=(others=>'0')",
            }
        )
        for i in range(len(expected_inputs_lines)):
            self.assertEqual(
                expected_inputs_lines[i], signal_definition_string.splitlines()[i]
            )

    def test_generate_libraries_with_math_lib(self) -> None:
        expected_import_lines = [
            "library ieee;",
            "use ieee.std_logic_1164.all;",
            "use ieee.numeric_std.all;               -- for type conversions",
            "use ieee.math_real.all;",
        ]
        lib_string = get_libraries_string(math_lib=True)
        for i in range(0, 4):
            self.assertEqual(expected_import_lines[i], lib_string.splitlines()[i])

    def test_generate_variable_definition(self) -> None:
        expected_variable_lines = ["    clk <= clock;"]
        variable_definition_string = get_variable_definitions_string(
            variable_dict={
                "clk": "clock",
            }
        )
        for i in range(len(expected_variable_lines)):
            self.assertEqual(
                expected_variable_lines[i], variable_definition_string.splitlines()[i]
            )

    def test_generate_architecture_header(self) -> None:
        expected_architecture_header_line = "architecture behav of sigmoid_tb is"
        architecture_header_string = get_architecture_header_string(
            architecture_name="behav", component_name="sigmoid_tb"
        )
        self.assertEqual(
            expected_architecture_header_line,
            architecture_header_string.splitlines()[0],
        )

    def test_generate_architecture_end(self) -> None:
        expected_test_architecture_end_line = "end architecture behav ; -- behav"
        architecture_end_string = get_architecture_end_string(architecture_name="behav")
        self.assertEqual(
            expected_test_architecture_end_line, architecture_end_string.splitlines()[0]
        )
