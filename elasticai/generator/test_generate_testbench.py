import unittest
from os.path import exists
from elasticai.generator.generate_testbench import main, write_test_process


class GenerateTestBenchTest(unittest.TestCase):
    def setUp(self) -> None:
        main()
        self.generated_testbench_file = open('../testbench/generated_sigmoid_tb.vhd', 'r')
        self.lines = self.generated_testbench_file.readlines()

    def tearDown(self) -> None:
        self.generated_testbench_file.close()

    def test_generate_file(self) -> None:
        self.assertTrue(exists('../testbench/sigmoid_generate_tb.vhd'))

    def test_generate_libraries(self) -> None:
        expected_import_lines = [
            "library ieee;\n",
            "use ieee.std_logic_1164.all;\n",
            "use ieee.numeric_std.all;               -- for type conversions\n"
        ]
        for i in range(0, 3):
            self.assertEqual(expected_import_lines[i], self.lines[i])

    def test_generate_entity(self) -> None:
        expected_entity_lines = [
            "entity sigmoid_tb is\n",
            "    port ( clk: out std_logic);\n",
            "end entity ; -- sigmoid_tb\n"
        ]
        for i, j in zip(range(0, 3), range(4, 7)):
            self.assertEqual(expected_entity_lines[i], self.lines[j])

    def test_generate_architecture_header(self) -> None:
        expected_architecture_header_lines = [
            "architecture behav of sigmoid_tb is\n",
        ]
        self.assertEqual(expected_architecture_header_lines[0], self.lines[8])

    def test_generate_component(self) -> None:
        expected_component_lines = [
            "    component sigmoid is\n",
            "        generic (\n",
            "                DATA_WIDTH : integer := 16;\n",
            "                FRAC_WIDTH : integer := 8\n",
            "            );\n",
            "        port (\n",
            "            x : in signed(DATA_WIDTH-1 downto 0);\n",
            "            y: out signed(DATA_WIDTH-1 downto 0)\n",
            "        );\n",
            "    end component;\n"
        ]
        for i, j in zip(range(len(expected_component_lines)), range(10, 20)):
            self.assertEqual(expected_component_lines[i], self.lines[j])

    def test_generate_signal_definitions(self) -> None:
        expected_inputs_lines = [
            "    ------------------------------------------------------------\n",
            "    -- Testbench Internal Signals\n",
            "    ------------------------------------------------------------\n",
            "    signal clk_period : time := 1 ns;\n",
            "    signal test_input : signed(16-1 downto 0):=(others=>'0');\n",
            "    signal test_output : signed(16-1 downto 0);\n"
        ]
        for i, j in zip(range(len(expected_inputs_lines)), range(21, 27)):
            self.assertEqual(expected_inputs_lines[i], self.lines[j])

    def test_generate_clock_process(self) -> None:
        expected_clock_lines = [
            "begin\n",
            "\n",
            "    clock_process : process\n",
            "    begin\n",
            "        clk <= '0';\n",
            "        wait for clk_period/2;\n",
            "        clk <= '1';\n",
            "        wait for clk_period/2;\n",
            "    end process; -- clock_process\n"
        ]
        for i, j in zip(range(len(expected_clock_lines)), range(28, 37)):
            self.assertEqual(expected_clock_lines[i], self.lines[j])

    def test_generate_utt(self) -> None:
        expected_utt_lines = [
            "    utt: sigmoid\n",
            "    port map (\n",
            "    x => test_input,\n",
            "    y => test_output\n",
            "    );\n"
        ]
        for i, j in zip(range(len(expected_utt_lines)), range(38, 43)):
            self.assertEqual(expected_utt_lines[i], self.lines[j])

    def test_generate_test_process_header(self) -> None:
        expected_test_process_header_lines = [
            "test_process: process is\n",
            "    begin\n",
            "        Report \"======Simulation start======\" severity Note;\n",
        ]
        for i, j in zip(range(len(expected_test_process_header_lines)), range(44, 75)):
            self.assertEqual(expected_test_process_header_lines[i], self.lines[j])

    def test_generate_test_process(self) -> None:
        expected_test_process_lines = [
            "        test_input <=  to_signed(-1281,16);\n",
            "        wait for 1*clk_period;\n",
            "        report \"The value of 'test_output' is \" & integer'image(to_integer(unsigned(test_output)));\n",
            "        assert test_output=0 report \"The test case -1281 fail\" severity failure;\n",
            "\n",
            "        test_input <=  to_signed(-1000,16);\n",
            "        wait for 1*clk_period;\n",
            "        report \"The value of 'test_output' is \" & integer'image(to_integer(unsigned(test_output)));\n",
            "        assert test_output=4 report \"The test case -1000 fail\" severity failure;\n",
            "\n",
            "        test_input <=  to_signed(-500,16);\n",
            "        wait for 1*clk_period;\n",
            "        report \"The value of 'test_output' is \" & integer'image(to_integer(unsigned(test_output)));\n",
            "        assert test_output=28 report \"The test case -500 fail\" severity failure;\n",
        ]
        for i, j in zip(range(len(expected_test_process_lines)), range(48, 62)):
            self.assertEqual(expected_test_process_lines[i], self.lines[j])

    def test_generate_test_process_raises_error_when_called_with_different_inputs_and_outputs_lenghts(self) -> None:
        self.assertRaises(TypeError, write_test_process, [1], [])

    def test_generate_test_process_end(self) -> None:
        expected_test_process_end_lines = [
            "        -- if there is no error message, that means all test case are passed.\n",
            "        report \"======Simulation Success======\" severity Note;\n",
            "        report \"Please check the output message.\" severity Note;\n",
            "        \n",
            "        -- wait forever\n",
            "        wait;\n",
            "        \n",
            "    end process; -- test_process\n",
        ]
        for i, j in zip(range(len(expected_test_process_end_lines)), range(64, 72)):
            self.assertEqual(expected_test_process_end_lines[i], self.lines[j])

    def test_generate_architecture_end(self) -> None:
        expected_test_architecture_end_lines = [
            "end behav ; -- behav\n"
        ]
        self.assertEqual(expected_test_architecture_end_lines[0], self.lines[73])
