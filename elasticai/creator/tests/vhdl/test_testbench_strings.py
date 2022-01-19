import unittest
from elasticai.creator.vhdl.generator.testbench_strings import (
    get_type_definitions_string,
    get_clock_process_string,
    get_uut_string,
    get_test_process_header_string,
    get_test_process_end_string,
)


class TestbenchStringsTest(unittest.TestCase):
    def test_generate_type_definition(self) -> None:
        expected_inputs_lines = [
            "\t------------------------------------------------------------",
            "\t-- Testbench Data Type",
            "\t------------------------------------------------------------",
            "\ttype RAM_ARRAY is array (0 to 9 ) of signed(DATA_WIDTH-1 downto 0);",
        ]
        type_definition_string = get_type_definitions_string(
            type_dict={
                "RAM_ARRAY": "array (0 to 9 ) of signed(DATA_WIDTH-1 downto 0)",
            }
        )
        for i in range(len(expected_inputs_lines)):
            self.assertEqual(
                expected_inputs_lines[i], type_definition_string.splitlines()[i]
            )

    def test_generate_clock_process(self) -> None:
        expected_clock_lines = [
            "\tclock_process : process",
            "\tbegin",
            "\t\tclk <= '0';",
            "\t\twait for clk_period/2;",
            "\t\tclk <= '1';",
            "\t\twait for clk_period/2;",
            "\tend process; -- clock_process",
        ]
        clock_process_string = get_clock_process_string()
        for i in range(len(expected_clock_lines)):
            self.assertEqual(
                expected_clock_lines[i], clock_process_string.splitlines()[i]
            )

    def test_generate_uut(self) -> None:
        expected_uut_lines = [
            "\tuut: sigmoid",
            "\tport map (",
            "\t\tx => test_input,",
            "\t\ty => test_output",
            "\t);",
        ]
        utt_string = get_uut_string(
            component_name="sigmoid",
            mapping_dict={"x": "test_input", "y": "test_output"},
        )
        for i in range(len(expected_uut_lines)):
            self.assertEqual(expected_uut_lines[i], utt_string.splitlines()[i])

    def test_generate_bigger_uut(self) -> None:
        expected_uut_lines = [
            "\tuut: lstm_common_gate",
            "\tport map (",
            "\t\treset => reset,",
            "\t\tclk => clock,",
            "\t\tx => x,",
            "\t\tw => w,",
            "\t\tb => b,",
            "\t\tvector_len => vector_len,",
            "\t\tidx => idx,",
            "\t\tready => ready,",
            "\t\ty => y",
            "\t);",
        ]
        utt_string = get_uut_string(
            component_name="lstm_common_gate",
            mapping_dict={
                "reset": "reset",
                "clk": "clock",
                "x": "x",
                "w": "w",
                "b": "b",
                "vector_len": "vector_len",
                "idx": "idx",
                "ready": "ready",
                "y": "y",
            },
        )
        for i in range(len(expected_uut_lines)):
            self.assertEqual(expected_uut_lines[i], utt_string.splitlines()[i])

    def test_generate_test_process_header(self) -> None:
        expected_test_process_header_lines = [
            "\ttest_process: process is",
            "\tbegin",
            '\t\tReport "======Simulation start======" severity Note;',
        ]
        test_process_header_string = get_test_process_header_string()
        for i in range(len(expected_test_process_header_lines)):
            self.assertEqual(
                expected_test_process_header_lines[i],
                test_process_header_string.splitlines()[i],
            )

    def test_generate_test_process_end(self) -> None:
        expected_test_process_end_lines = [
            "",
            "\t\t-- if there is no error message, that means all test case are passed.",
            '\t\treport "======Simulation Success======" severity Note;',
            '\t\treport "Please check the output message." severity Note;',
            "",
            "\t\t-- wait forever",
            "\t\twait;",
            "",
            "\tend process; -- test_process",
        ]
        test_process_end_string = get_test_process_end_string()
        for i in range(len(expected_test_process_end_lines)):
            self.assertEqual(
                expected_test_process_end_lines[i],
                test_process_end_string.splitlines()[i],
            )
