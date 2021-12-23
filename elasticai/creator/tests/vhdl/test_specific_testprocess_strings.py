import unittest
from elasticai.creator.vhdl.generator.specific_testprocess_strings import (
    get_test_process_for_one_input_results_in_one_output_string,
    get_test_process_for_multiple_input_results_in_one_output_string,
)


class SpecificTestProcessStringsTest(unittest.TestCase):
    def test_generate_test_process_for_one_input(self) -> None:
        expected_test_process_lines = [
            "        test_input <=  to_signed(-1281,16);",
            "        wait for 1*clk_period;",
            "        report \"The value of 'test_output' is \" & integer'image(to_integer(unsigned(test_output)));",
            '        assert test_output=0 report "The test case -1281 fail" severity failure;',
            "",
            "        test_input <=  to_signed(-1000,16);",
            "        wait for 1*clk_period;",
            "        report \"The value of 'test_output' is \" & integer'image(to_integer(unsigned(test_output)));",
            '        assert test_output=4 report "The test case -1000 fail" severity failure;',
            "",
            "        test_input <=  to_signed(-500,16);",
            "        wait for 1*clk_period;",
            "        report \"The value of 'test_output' is \" & integer'image(to_integer(unsigned(test_output)));",
            '        assert test_output=28 report "The test case -500 fail" severity failure;',
        ]
        test_process_string = (
            get_test_process_for_one_input_results_in_one_output_string(
                inputs=[-1281, -1000, -500],
                outputs=[0, 4, 28],
                input_name="test_input",
                output_name="test_output",
            )
        )
        for i in range(len(expected_test_process_lines)):
            self.assertEqual(
                expected_test_process_lines[i], test_process_string.splitlines()[i]
            )

    def test_generate_test_process_raises_error_when_called_with_different_inputs_and_outputs_lenghts(
        self,
    ) -> None:
        self.assertRaises(
            TypeError,
            get_test_process_for_one_input_results_in_one_output_string,
            [1],
            [],
        )

    def test_generate_test_process_for_multiple_inputs(self) -> None:
        expected_test_process_lines = [
            '        X_MEM <= (x"0013",x"0000",x"0010",x"0013",x"000c",x"0005",x"0005",x"0013",x"0004",x"0002");',
            '        W_MEM <= (x"0011",x"0018",x"0000",x"000d",x"0014",x"000f",x"0012",x"0007",x"0017",x"0012");',
            '        b <= x"008a";',
            "",
            "        reset <= '1';",
            "        wait for 2*clk_period;",
            "        wait until clock = '0';",
            "        reset <= '0';",
            "        wait until ready = '1';",
            "",
            "        report \"expected output is 142, value of 'y' is \" & integer'image(to_integer(signed(y)));",
            '        assert y = 142 report "The 0. test case fail" severity error;',
            "        reset <= '1';",
            "        wait for 1*clk_period;",
            "",
            '        X_MEM <= (x"0014",x"000d",x"0017",x"0008",x"0002",x"0007",x"0002",x"0015",x"0001",x"0010");',
            '        W_MEM <= (x"000e",x"0014",x"0005",x"0015",x"0009",x"0013",x"0007",x"0016",x"0008",x"0004");',
            '        b <= x"0064";',
            "",
            "        reset <= '1';",
            "        wait for 2*clk_period;",
            "        wait until clock = '0';",
            "        reset <= '0';",
            "        wait until ready = '1';",
            "",
            "        report \"expected output is 105, value of 'y' is \" & integer'image(to_integer(signed(y)));",
            '        assert y = 105 report "The 1. test case fail" severity error;',
            "        reset <= '1';",
            "        wait for 1*clk_period;",
            "",
            '        X_MEM <= (x"000f",x"0017",x"000d",x"000f",x"0001",x"0009",x"0002",x"0007",x"0008",x"0013");',
            '        W_MEM <= (x"0001",x"000a",x"0008",x"0010",x"0008",x"0001",x"0016",x"0013",x"0016",x"000a");',
            '        b <= x"009b";',
            "",
            "        reset <= '1';",
            "        wait for 2*clk_period;",
            "        wait until clock = '0';",
            "        reset <= '0';",
            "        wait until ready = '1';",
            "",
            "        report \"expected output is 159, value of 'y' is \" & integer'image(to_integer(signed(y)));",
            '        assert y = 159 report "The 2. test case fail" severity error;',
            "        reset <= '1';",
            "        wait for 1*clk_period;",
            "",
            '        X_MEM <= (x"000c",x"0007",x"0001",x"0019",x"0008",x"000c",x"0019",x"000b",x"0008",x"000d");',
            '        W_MEM <= (x"000e",x"0015",x"0001",x"000b",x"0014",x"0012",x"000f",x"0000",x"0008",x"000e");',
            '        b <= x"004c";',
            "",
            "        reset <= '1';",
            "        wait for 2*clk_period;",
            "        wait until clock = '0';",
            "        reset <= '0';",
            "        wait until ready = '1';",
            "",
            "        report \"expected output is 82, value of 'y' is \" & integer'image(to_integer(signed(y)));",
            '        assert y = 82 report "The 3. test case fail" severity error;',
            "        reset <= '1';",
            "        wait for 1*clk_period;",
            "",
            '        X_MEM <= (x"0005",x"0013",x"0002",x"0013",x"000c",x"000f",x"0003",x"0004",x"0010",x"0001");',
            '        W_MEM <= (x"0006",x"000d",x"0005",x"0009",x"0017",x"0017",x"000e",x"000d",x"0000",x"0019");',
            '        b <= x"0092";',
            "",
            "        reset <= '1';",
            "        wait for 2*clk_period;",
            "        wait until clock = '0';",
            "        reset <= '0';",
            "        wait until ready = '1';",
            "",
            "        report \"expected output is 150, value of 'y' is \" & integer'image(to_integer(signed(y)));",
            '        assert y = 150 report "The 4. test case fail" severity error;',
            "        reset <= '1';",
            "        wait for 1*clk_period;",
        ]
        inputs = [
            {
                "X_MEM": '(x"0013",x"0000",x"0010",x"0013",x"000c",x"0005",x"0005",x"0013",x"0004",x"0002")',
                "W_MEM": '(x"0011",x"0018",x"0000",x"000d",x"0014",x"000f",x"0012",x"0007",x"0017",x"0012")',
                "b": 'x"008a"',
            },
            {
                "X_MEM": '(x"0014",x"000d",x"0017",x"0008",x"0002",x"0007",x"0002",x"0015",x"0001",x"0010")',
                "W_MEM": '(x"000e",x"0014",x"0005",x"0015",x"0009",x"0013",x"0007",x"0016",x"0008",x"0004")',
                "b": 'x"0064"',
            },
            {
                "X_MEM": '(x"000f",x"0017",x"000d",x"000f",x"0001",x"0009",x"0002",x"0007",x"0008",x"0013")',
                "W_MEM": '(x"0001",x"000a",x"0008",x"0010",x"0008",x"0001",x"0016",x"0013",x"0016",x"000a")',
                "b": 'x"009b"',
            },
            {
                "X_MEM": '(x"000c",x"0007",x"0001",x"0019",x"0008",x"000c",x"0019",x"000b",x"0008",x"000d")',
                "W_MEM": '(x"000e",x"0015",x"0001",x"000b",x"0014",x"0012",x"000f",x"0000",x"0008",x"000e")',
                "b": 'x"004c"',
            },
            {
                "X_MEM": '(x"0005",x"0013",x"0002",x"0013",x"000c",x"000f",x"0003",x"0004",x"0010",x"0001")',
                "W_MEM": '(x"0006",x"000d",x"0005",x"0009",x"0017",x"0017",x"000e",x"000d",x"0000",x"0019")',
                "b": 'x"0092"',
            },
        ]
        # expected signal, as test reference output signal
        outputs = [142, 105, 159, 82, 150]
        test_process_string = (
            get_test_process_for_multiple_input_results_in_one_output_string(
                inputs=inputs, outputs=outputs, output_name="y"
            )
        )
        for i in range(len(expected_test_process_lines)):
            self.assertEqual(
                expected_test_process_lines[i], test_process_string.splitlines()[i]
            )

    def test_generate_test_process_multiple_inputs_raises_error_when_called_with_different_inputs_and_outputs_lenghts(
        self,
    ) -> None:
        self.assertRaises(
            TypeError,
            get_test_process_for_multiple_input_results_in_one_output_string,
            [1],
            [],
            "y",
        )
