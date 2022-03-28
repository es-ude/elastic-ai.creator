from elasticai.creator.vhdl.language_testbench import (
    TestCasesPrecomputedScalarFunction,
    TestCasesLSTMCommonGate,
    TestCasesLSTMCell,
    Procedure,
)
from unittest import TestCase


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


class TestCasesLSTMCellTest(TestCase):
    def test_TestCasesLSTMCell(self):
        reference_h_out = [1, 2, 3, 4, -5]
        test_cases_lstm_cell = TestCasesLSTMCell(reference_h_out=reference_h_out)
        expected = [
            f'report "======Tests Start======" severity Note;',
            f"reset <= '1';",
            f"h_out_en <= '0';",
            f"wait for 2*clk_period;",
            f"reset <= '0';",
            f"for ii in 0 to 24 loop",
            f"send_x_h_data(std_logic_vector(to_unsigned(ii, X_H_ADDR_WIDTH)), std_logic_vector(test_x_h_data(ii)), clock, x_config_en, x_config_addr, x_config_data);",
            f"wait for 10 ns;",
            f"end loop;",
            f"for ii in 0 to 19 loop",
            f"send_c_data(std_logic_vector(to_unsigned(ii, HIDDEN_ADDR_WIDTH)), std_logic_vector(test_c_data(ii)), clock, c_config_en, c_config_addr, c_config_data);",
            f"wait for 10 ns;",
            f"end loop;",
            f"enable <= '1';",
            f"wait until done = '1';",
            f"wait for 1*clk_period;",
            f"enable <= '0';",
            f"-- reference h_out: {str(reference_h_out)}",
            f"for ii in 0 to 19 loop",
            f"h_out_addr <= std_logic_vector(to_unsigned(ii, HIDDEN_ADDR_WIDTH));",
            f"h_out_en <= '1';",
            f"wait for 2*clk_period;",
            f'report "The value of h_out(" & integer\'image(ii)& ") is " & integer\'image(to_integer(signed(h_out_data)));',
            f"end loop;",
            f"wait for 10*clk_period;",
            f'report "======Tests finished======" severity Note;',
            f'report "Please check the output message." severity Note;',
            f"wait;",
        ]
        actual = list(test_cases_lstm_cell())
        self.assertSequenceEqual(expected, actual)


class TestCasesLSTMCommonGateTest(TestCase):
    def test_TestCasesLSTMCommonGate(self):
        x_mem_list_for_testing = [
            "some_string_00",
            "some_string_01",
            "some_string_02",
        ]
        w_mem_list_for_testing = [
            "some_string_10",
            "some_string_11",
            "some_string_12",
        ]
        b_list_for_testing = ["some_string_20", "some_string_21", "some_string_22"]
        y_list_for_testing = [1, 2, 3]
        y_variable_name = "out"
        test_cases_lstm_common_gate = TestCasesLSTMCommonGate(
            x_mem_list_for_testing=x_mem_list_for_testing,
            w_mem_list_for_testing=w_mem_list_for_testing,
            b_list_for_testing=b_list_for_testing,
            y_list_for_testing=y_list_for_testing,
            y_variable_name=y_variable_name,
        )
        expected = [
            f'report "======Simulation Start======" severity Note;',
            f"vector_len <= to_unsigned(10, VECTOR_LEN_WIDTH);",
            f"X_MEM <= ({x_mem_list_for_testing[0]});",
            f"W_MEM <= ({w_mem_list_for_testing[0]});",
            f"b <= {b_list_for_testing[0]};",
            f"reset <= '1';",
            f"wait for 2*clk_period;",
            f"wait until clock = '0';",
            f"reset <= '0';",
            f"wait until ready = '1';",
            f"report \"expected output is {y_list_for_testing[0]}, value of '{y_variable_name}' is \" & integer'image(to_integer(signed({y_variable_name})));",
            f'assert {y_variable_name}={y_list_for_testing[0]} report "The 0. test case fail" severity error;',
            f"reset <= '1';",
            f"wait for 1*clk_period;",
            f"X_MEM <= ({x_mem_list_for_testing[1]});",
            f"W_MEM <= ({w_mem_list_for_testing[1]});",
            f"b <= {b_list_for_testing[1]};",
            f"reset <= '1';",
            f"wait for 2*clk_period;",
            f"wait until clock = '0';",
            f"reset <= '0';",
            f"wait until ready = '1';",
            f"report \"expected output is {y_list_for_testing[1]}, value of '{y_variable_name}' is \" & integer'image(to_integer(signed({y_variable_name})));",
            f'assert {y_variable_name}={y_list_for_testing[1]} report "The 1. test case fail" severity error;',
            f"reset <= '1';",
            f"wait for 1*clk_period;",
            f"X_MEM <= ({x_mem_list_for_testing[2]});",
            f"W_MEM <= ({w_mem_list_for_testing[2]});",
            f"b <= {b_list_for_testing[2]};",
            f"reset <= '1';",
            f"wait for 2*clk_period;",
            f"wait until clock = '0';",
            f"reset <= '0';",
            f"wait until ready = '1';",
            f"report \"expected output is {y_list_for_testing[2]}, value of '{y_variable_name}' is \" & integer'image(to_integer(signed({y_variable_name})));",
            f'assert {y_variable_name}={y_list_for_testing[2]} report "The 2. test case fail" severity error;',
            f"reset <= '1';",
            f"wait for 1*clk_period;",
            f'report "======Simulation Success======" severity Note;',
            f'report "Please check the output message." severity Note;',
            f"wait;",
        ]
        actual = list(test_cases_lstm_common_gate())
        self.assertSequenceEqual(expected, actual)

    def test_TestCasesLSTMCommonGate_different_lengths_of_lists(self):
        x_mem_list_for_testing = [
            "some_string_00",
            "some_string_01",
        ]
        w_mem_list_for_testing = [
            "some_string_10",
            "some_string_11",
            "some_string_12",
            "some_string_13",
        ]
        b_list_for_testing = ["some_string_20", "some_string_21", "some_string_22"]
        y_list_for_testing = [1, 2, 3]
        y_variable_name = "out"
        self.assertRaises(
            AssertionError,
            TestCasesLSTMCommonGate,
            x_mem_list_for_testing,
            w_mem_list_for_testing,
            b_list_for_testing,
            y_list_for_testing,
            y_variable_name,
        )


class TestCasesPrecomputedScalarFunctionTest(TestCase):
    def test_TestCasesPrecomputedScalarFunction(self):
        x_list_for_testing = [1, 2, 3]
        y_list_for_testing = [-5, "some_string", -3]
        x_variable_name = "x_name"
        y_variable_name = "y_name"
        data_width = 3
        test_cases_precomputed_scalar_function = TestCasesPrecomputedScalarFunction(
            x_list_for_testing=x_list_for_testing,
            y_list_for_testing=y_list_for_testing,
            x_variable_name=x_variable_name,
            y_variable_name=y_variable_name,
            data_width=data_width,
        )
        expected = [
            f'report "======Simulation Start======" severity Note;',
            f"{x_variable_name} <= to_signed({x_list_for_testing[0]},{data_width});",
            f"wait for 1*clk_period;",
            f"report \"The value of '{y_variable_name}' is \" & integer'image(to_integer(unsigned({y_variable_name})));",
            f'assert {y_variable_name}={y_list_for_testing[0]} report "The test case {x_list_for_testing[0]} fail" severity failure;',
            f"{x_variable_name} <= to_signed({x_list_for_testing[1]},{data_width});",
            f"wait for 1*clk_period;",
            f"report \"The value of '{y_variable_name}' is \" & integer'image(to_integer(unsigned({y_variable_name})));",
            f'assert {y_variable_name}="{y_list_for_testing[1]}" report "The test case {x_list_for_testing[1]} fail" severity failure;',
            f"{x_variable_name} <= to_signed({x_list_for_testing[2]},{data_width});",
            f"wait for 1*clk_period;",
            f"report \"The value of '{y_variable_name}' is \" & integer'image(to_integer(unsigned({y_variable_name})));",
            f'assert {y_variable_name}={y_list_for_testing[2]} report "The test case {x_list_for_testing[2]} fail" severity failure;',
            f'report "======Simulation Success======" severity Note;',
            f'report "Please check the output message." severity Note;',
            f"wait;",
        ]
        actual = list(test_cases_precomputed_scalar_function())
        self.assertSequenceEqual(expected, actual)

    def test_TestCasesPrecomputedScalarFunction_different_lenghts_of_list(self):
        x_list_for_testing = [1, 2, 3]
        y_list_for_testing = [-5, -3]
        x_variable_name = "x_name"
        y_variable_name = "y_name"
        data_width = 3
        self.assertRaises(
            AssertionError,
            TestCasesPrecomputedScalarFunction,
            x_list_for_testing,
            y_list_for_testing,
            x_variable_name,
            y_variable_name,
            data_width,
        )
