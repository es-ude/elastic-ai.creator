from functools import partial
from unittest import TestCase

from elasticai.creator.vhdl.language import hex_representation
from elasticai.creator.vhdl.lstm_testbench_generator import TestCasesLSTMCommonGate
from elasticai.creator.vhdl.number_representations import (
    FixedPoint,
    float_values_to_fixed_point,
)
from elasticai.creator.vhdl.precomputed_scalar_function import (
    TestCasesPrecomputedScalarFunction,
)


class TestCasesLSTMCommonGateTest(TestCase):
    def test_TestCasesLSTMCommonGate(self):
        to_fp = partial(FixedPoint, total_bits=8, frac_bits=0)

        def to_hex(value: FixedPoint) -> str:
            return hex_representation(value.to_hex())

        x_mem_list_for_testing = [to_fp(x) for x in range(0, 3)]
        w_mem_list_for_testing = [to_fp(x) for x in range(10, 13)]
        b_list_for_testing = [to_fp(x) for x in range(20, 23)]
        y_list_for_testing = [to_fp(x) for x in range(30, 33)]
        y_variable_name = "out"
        test_cases_lstm_common_gate = TestCasesLSTMCommonGate(
            x_mem_list_for_testing=x_mem_list_for_testing,
            w_mem_list_for_testing=w_mem_list_for_testing,
            b_list_for_testing=b_list_for_testing,
            y_list_for_testing=y_list_for_testing,
            y_variable_name=y_variable_name,
        )
        expected = [
            f'report "======Simulation Start======" severity Note',
            f"vector_len <= to_unsigned(10, VECTOR_LEN_WIDTH)",
            f"X_MEM <= ({to_hex(x_mem_list_for_testing[0])})",
            f"W_MEM <= ({to_hex(w_mem_list_for_testing[0])})",
            f"b <= {to_hex(b_list_for_testing[0])}",
            f"reset <= '1'",
            f"wait for 2*clk_period",
            f"wait until clock = '0'",
            f"reset <= '0'",
            f"wait until ready = '1'",
            f"report \"expected output is {y_list_for_testing[0]}, value of '{y_variable_name}' is \" & integer'image(to_integer(signed({y_variable_name})))",
            f'assert {y_variable_name}={y_list_for_testing[0]} report "The 0. test case fail" severity error',
            f"reset <= '1'",
            f"wait for 1*clk_period",
            f"X_MEM <= ({to_hex(x_mem_list_for_testing[1])})",
            f"W_MEM <= ({to_hex(w_mem_list_for_testing[1])})",
            f"b <= {to_hex(b_list_for_testing[1])}",
            f"reset <= '1'",
            f"wait for 2*clk_period",
            f"wait until clock = '0'",
            f"reset <= '0'",
            f"wait until ready = '1'",
            f"report \"expected output is {y_list_for_testing[1]}, value of '{y_variable_name}' is \" & integer'image(to_integer(signed({y_variable_name})))",
            f'assert {y_variable_name}={y_list_for_testing[1]} report "The 1. test case fail" severity error',
            f"reset <= '1'",
            f"wait for 1*clk_period",
            f"X_MEM <= ({to_hex(x_mem_list_for_testing[2])})",
            f"W_MEM <= ({to_hex(w_mem_list_for_testing[2])})",
            f"b <= {to_hex(b_list_for_testing[2])}",
            f"reset <= '1'",
            f"wait for 2*clk_period",
            f"wait until clock = '0'",
            f"reset <= '0'",
            f"wait until ready = '1'",
            f"report \"expected output is {y_list_for_testing[2]}, value of '{y_variable_name}' is \" & integer'image(to_integer(signed({y_variable_name})))",
            f'assert {y_variable_name}={y_list_for_testing[2]} report "The 2. test case fail" severity error',
            f"reset <= '1'",
            f"wait for 1*clk_period",
            f'report "======Simulation Success======" severity Note',
            f'report "Please check the output message." severity Note',
            f"wait",
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
    def setUp(self) -> None:
        self.data_width, self.frac_width = 8, 0
        self.fp_list = partial(
            float_values_to_fixed_point,
            total_bits=self.data_width,
            frac_bits=self.frac_width,
        )

    def test_TestCasesPrecomputedScalarFunction(self):
        x_list_for_testing = [1, 2, 3]
        y_list_for_testing = [-5, -2, -3]
        x_variable_name = "x_name"
        y_variable_name = "y_name"
        test_cases_precomputed_scalar_function = TestCasesPrecomputedScalarFunction(
            x_list_for_testing=self.fp_list(x_list_for_testing),
            y_list_for_testing=self.fp_list(y_list_for_testing),
            x_variable_name=x_variable_name,
            y_variable_name=y_variable_name,
        )
        expected = [
            f'report "======Simulation Start======" severity Note',
            f"{x_variable_name} <= to_signed({x_list_for_testing[0]},{self.data_width})",
            f"wait for 1*clk_period",
            f"report \"The value of '{y_variable_name}' is \" & integer'image(to_integer(unsigned({y_variable_name})))",
            f'assert {y_variable_name}={y_list_for_testing[0]} report "The test case {x_list_for_testing[0]} fail" severity failure',
            f"{x_variable_name} <= to_signed({x_list_for_testing[1]},{self.data_width})",
            f"wait for 1*clk_period",
            f"report \"The value of '{y_variable_name}' is \" & integer'image(to_integer(unsigned({y_variable_name})))",
            f'assert {y_variable_name}={y_list_for_testing[1]} report "The test case {x_list_for_testing[1]} fail" severity failure',
            f"{x_variable_name} <= to_signed({x_list_for_testing[2]},{self.data_width})",
            f"wait for 1*clk_period",
            f"report \"The value of '{y_variable_name}' is \" & integer'image(to_integer(unsigned({y_variable_name})))",
            f'assert {y_variable_name}={y_list_for_testing[2]} report "The test case {x_list_for_testing[2]} fail" severity failure',
            f'report "======Simulation Success======" severity Note',
            f'report "Please check the output message." severity Note',
            f"wait",
        ]
        actual = list(test_cases_precomputed_scalar_function())
        self.assertSequenceEqual(expected, actual)

    def test_TestCasesPrecomputedScalarFunction_different_lengths_of_list(self):
        x_list_for_testing = [1, 2, 3]
        y_list_for_testing = [-5, -3]
        x_variable_name = "x_name"
        y_variable_name = "y_name"
        self.assertRaises(
            AssertionError,
            TestCasesPrecomputedScalarFunction,
            self.fp_list(x_list_for_testing),
            self.fp_list(y_list_for_testing),
            x_variable_name,
            y_variable_name,
        )
