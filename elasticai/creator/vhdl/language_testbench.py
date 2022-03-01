"""
The module contains classes and functions for generating vhdl code similar to the language module
This module includes CodeGenerator that are only used by the vhdl testbenches
"""
from enum import Enum
from elasticai.creator.vhdl.language import _indent_and_filter_non_empty_lines


class Keywords(Enum):
    wait = "wait"


class TestCases:
    def __init__(self, x_list_for_testing, y_list_for_testing):
        assert len(x_list_for_testing) == len(y_list_for_testing)
        self.x_list_for_testing = x_list_for_testing
        self.y_list_for_testing = y_list_for_testing

    def __call__(self):
        yield f'report "======Simulation Start======" severity Note;'
        for x_value, y_value in zip(self.x_list_for_testing, self.y_list_for_testing):
            yield f"test_input <= to_signed({x_value},16);"
            yield f"wait for 1*clk_period;"
            yield f"report \"The value of 'test_output' is \" & integer'image(to_integer(unsigned(test_output)));"
            yield f'assert test_output={y_value} report "The test case {x_value} fail" severity failure;'
        yield f'report "======Simulation Success======" severity Note;'
        yield f'report "Please check the output message." severity Note;'
        yield f"wait;"
