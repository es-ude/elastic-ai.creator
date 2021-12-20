import unittest
from elasticai.generator.generate_specific_testprocess import get_test_process_for_one_input_results_in_one_output_string


class GenerateSpecificTestProcessTest(unittest.TestCase):
    def test_generate_test_process(self) -> None:
        expected_test_process_lines = [
            "        test_input <=  to_signed(-1281,16);",
            "        wait for 1*clk_period;",
            "        report \"The value of 'test_output' is \" & integer'image(to_integer(unsigned(test_output)));",
            "        assert test_output=0 report \"The test case -1281 fail\" severity failure;",
            "",
            "        test_input <=  to_signed(-1000,16);",
            "        wait for 1*clk_period;",
            "        report \"The value of 'test_output' is \" & integer'image(to_integer(unsigned(test_output)));",
            "        assert test_output=4 report \"The test case -1000 fail\" severity failure;",
            "",
            "        test_input <=  to_signed(-500,16);",
            "        wait for 1*clk_period;",
            "        report \"The value of 'test_output' is \" & integer'image(to_integer(unsigned(test_output)));",
            "        assert test_output=28 report \"The test case -500 fail\" severity failure;",
        ]
        test_process_string = get_test_process_for_one_input_results_in_one_output_string(inputs=[-1281, -1000, -500], outputs=[0, 4, 28], input_name="test_input", output_name="test_output")
        for i in range(len(expected_test_process_lines)):
            self.assertEqual(expected_test_process_lines[i], test_process_string.splitlines()[i])

    def test_generate_test_process_raises_error_when_called_with_different_inputs_and_outputs_lenghts(self) -> None:
        self.assertRaises(TypeError, get_test_process_for_one_input_results_in_one_output_string, [1], [])
