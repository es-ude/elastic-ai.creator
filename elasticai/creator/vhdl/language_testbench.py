"""
The module contains classes and functions for generating vhdl code similar to the language module
This module includes CodeGenerator that are only used by the vhdl testbenches
"""
from enum import Enum


class Keywords(Enum):
    wait = "wait"
    report = "report"


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
            yield f'assert {self.y_variable_name}={y_value} {Keywords.report.value} "The test case {x_value} fail" severity failure;'
        yield f'{Keywords.report.value} "======Simulation Success======" severity Note;'
        yield f'{Keywords.report.value} "Please check the output message." severity Note;'
        yield f"{Keywords.wait.value};"
