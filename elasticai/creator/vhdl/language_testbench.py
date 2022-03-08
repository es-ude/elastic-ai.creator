"""
The module contains classes and functions for generating vhdl code similar to the language module
This module includes CodeGenerator that are only used by the vhdl testbenches
"""
from enum import Enum


class Keywords(Enum):
    wait = "wait"
    report = "report"
    reset = "reset"


class TestCases:
    def __init__(
        self,
        x_list_for_testing: list[int],
        y_list_for_testing: list[int],
        x_variable_name: str = "test_input",
        y_variable_name: str = "test_output",
        data_width: int = 16,
    ):
        assert len(x_list_for_testing) == len(y_list_for_testing)
        self.x_list_for_testing = x_list_for_testing
        self.y_list_for_testing = y_list_for_testing
        self.x_variable_name = x_variable_name
        self.y_variable_name = y_variable_name
        self.data_width = data_width

    def __call__(self):
        yield f'report "======Simulation Start======" severity Note;'
        for x_value, y_value in zip(self.x_list_for_testing, self.y_list_for_testing):
            yield f"{self.x_variable_name} <= to_signed({x_value},{self.data_width});"
            yield f"{Keywords.wait.value} for 1*clk_period;"
            yield f"{Keywords.report.value} \"The value of '{self.y_variable_name}' is \" & integer'image(to_integer(unsigned({self.y_variable_name})));"
            if isinstance(y_value, str):
                yield f'assert {self.y_variable_name}="{y_value}" {Keywords.report.value} "The test case {x_value} fail" severity failure;'
            else:
                yield f'assert {self.y_variable_name}={y_value} {Keywords.report.value} "The test case {x_value} fail" severity failure;'
        yield f'{Keywords.report.value} "======Simulation Success======" severity Note;'
        yield f'{Keywords.report.value} "Please check the output message." severity Note;'
        yield f"{Keywords.wait.value};"


class TestCasesLSTM:
    def __init__(
        self,
        x_mem_list_for_testing: list,
        w_mem_list_for_testing: list,
        b_list_for_testing: list,
        y_list_for_testing: list[int],
        y_variable_name: str = "y",
    ):
        assert (
            len(x_mem_list_for_testing)
            == len(w_mem_list_for_testing)
            == len(b_list_for_testing)
            == len(y_list_for_testing)
        )
        self.x_mem_list_for_testing = x_mem_list_for_testing
        self.w_mem_list_for_testing = w_mem_list_for_testing
        self.y_list_for_testing = y_list_for_testing
        self.b_list_for_testing = b_list_for_testing
        self.y_variable_name = y_variable_name

    def __call__(self):
        counter = 0
        yield f'report "======Simulation Start======" severity Note;'
        yield f"vector_len <= to_unsigned(10, VECTOR_LEN_WIDTH);"
        for x_mem_value, w_mem_value, b, y_value in zip(
            self.x_mem_list_for_testing,
            self.w_mem_list_for_testing,
            self.b_list_for_testing,
            self.y_list_for_testing,
        ):
            yield f"X_MEM <= ({x_mem_value});"
            yield f"W_MEM <= ({w_mem_value});"
            yield f"b <= {b};"
            yield f"{Keywords.reset.value} <= '1';"
            yield f"{Keywords.wait.value} for 2*clk_period;"
            yield f"{Keywords.wait.value} until clock = '0';"
            yield f"{Keywords.reset.value} <= '0';"
            yield f"{Keywords.wait.value} until ready = '1';"

            yield f"{Keywords.report.value} \"expected output is {y_value}, value of '{self.y_variable_name}' is \" & integer'image(to_integer(signed({self.y_variable_name})));"
            yield f'assert {self.y_variable_name}={y_value} {Keywords.report.value} "The {counter}. test case fail" severity error;'
            yield f"{Keywords.reset.value} <= '1';"
            yield f"{Keywords.wait.value} for 1*clk_period;"
            counter = counter + 1

        yield f'{Keywords.report.value} "======Simulation Success======" severity Note;'
        yield f'{Keywords.report.value} "Please check the output message." severity Note;'
        yield f"{Keywords.wait.value};"
