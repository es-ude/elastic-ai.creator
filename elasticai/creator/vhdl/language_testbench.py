"""
The module contains classes and functions for generating vhdl code similar to the language module
This module includes CodeGenerator that are only used by the vhdl testbenches
"""
from enum import Enum
from elasticai.creator.vhdl.language import (
    InterfaceList,
    _filter_empty_lines,
    _add_semicolons,
    _add_is,
    Keywords,
)


class KeywordsTestbench(Enum):
    WAIT = "wait"
    REPORT = "report"
    RESET = "reset"
    PROCEDURE = "procedure"
    FOR = "for"


class TestCasesProcomputedScalarFunction:
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
        yield f'{KeywordsTestbench.REPORT.value} "======Simulation Start======" severity Note;'
        for x_value, y_value in zip(self.x_list_for_testing, self.y_list_for_testing):
            yield f"{self.x_variable_name} <= to_signed({x_value},{self.data_width});"
            yield f"{KeywordsTestbench.WAIT.value} for 1*clk_period;"
            yield f"{KeywordsTestbench.REPORT.value} \"The value of '{self.y_variable_name}' is \" & integer'image(to_integer(unsigned({self.y_variable_name})));"
            if isinstance(y_value, str):
                yield f'assert {self.y_variable_name}="{y_value}" {KeywordsTestbench.REPORT.value} "The test case {x_value} fail" severity failure;'
            else:
                yield f'assert {self.y_variable_name}={y_value} {KeywordsTestbench.REPORT.value} "The test case {x_value} fail" severity failure;'
        yield f'{KeywordsTestbench.REPORT.value} "======Simulation Success======" severity Note;'
        yield f'{KeywordsTestbench.REPORT.value} "Please check the output message." severity Note;'
        yield f"{KeywordsTestbench.WAIT.value};"


class TestCasesLSTMCommonGate:
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
        yield f'{KeywordsTestbench.REPORT.value} "======Simulation Start======" severity Note;'
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
            yield f"{KeywordsTestbench.RESET.value} <= '1';"
            yield f"{KeywordsTestbench.WAIT.value} for 2*clk_period;"
            yield f"{KeywordsTestbench.WAIT.value} until clock = '0';"
            yield f"{KeywordsTestbench.RESET.value} <= '0';"
            yield f"{KeywordsTestbench.WAIT.value} until ready = '1';"

            yield f"{KeywordsTestbench.REPORT.value} \"expected output is {y_value}, value of '{self.y_variable_name}' is \" & integer'image(to_integer(signed({self.y_variable_name})));"
            yield f'assert {self.y_variable_name}={y_value} {KeywordsTestbench.REPORT.value} "The {counter}. test case fail" severity error;'
            yield f"{KeywordsTestbench.RESET.value} <= '1';"
            yield f"{KeywordsTestbench.WAIT.value} for 1*clk_period;"
            counter = counter + 1

        yield f'{KeywordsTestbench.REPORT.value} "======Simulation Success======" severity Note;'
        yield f'{KeywordsTestbench.REPORT.value} "Please check the output message." severity Note;'
        yield f"{KeywordsTestbench.WAIT.value};"


class TestCasesLSTMCell:
    def __init__(self, reference_h_out):
        self.reference_h_out = reference_h_out

    def __call__(self):
        yield f'{KeywordsTestbench.REPORT.value} "======Tests Start======" severity Note;'
        yield f"{KeywordsTestbench.RESET.value} <= '1';"
        yield f"h_out_en <= '0';"
        yield f"{KeywordsTestbench.WAIT.value} {KeywordsTestbench.FOR.value} 2*clk_period;"
        yield f"{KeywordsTestbench.RESET.value} <= '0';"
        yield f"{KeywordsTestbench.FOR.value} ii {Keywords.IN.value} 0 to 24 loop"
        yield f"send_x_h_data(std_logic_vector(to_unsigned(ii, X_H_ADDR_WIDTH)), std_logic_vector(test_x_h_data(ii)), clock, x_config_en, x_config_addr, x_config_data);"
        yield f"{KeywordsTestbench.WAIT.value} {KeywordsTestbench.FOR.value} 10 ns;"
        yield f"{Keywords.END.value} loop;"
        yield f"{KeywordsTestbench.FOR.value} ii {Keywords.IN.value} 0 to 19 loop"
        yield f"send_c_data(std_logic_vector(to_unsigned(ii, HIDDEN_ADDR_WIDTH)), std_logic_vector(test_c_data(ii)), clock, c_config_en, c_config_addr, c_config_data);"
        yield f"{KeywordsTestbench.WAIT.value} {KeywordsTestbench.FOR.value} 10 ns;"
        yield f"{Keywords.END.value} loop;"
        yield f"enable <= '1';"
        yield f"{KeywordsTestbench.WAIT.value} until done = '1';"
        yield f"{KeywordsTestbench.WAIT.value} {KeywordsTestbench.FOR.value} 1*clk_period;"
        yield f"enable <= '0';"
        yield f"-- reference h_out: {str(self.reference_h_out)}"
        yield f"{KeywordsTestbench.FOR.value} ii in 0 to 19 loop"
        yield f"h_out_addr <= std_logic_vector(to_unsigned(ii, HIDDEN_ADDR_WIDTH));"
        yield f"h_out_en <= '1';"
        yield f"{KeywordsTestbench.WAIT.value} {KeywordsTestbench.FOR.value} 2*clk_period;"
        yield f'{KeywordsTestbench.REPORT.value} "The value of h_out(" & integer\'image(ii)& ") is " & integer\'image(to_integer(signed(h_out_data)));'
        yield f"{Keywords.END.value} loop;"
        yield f"{KeywordsTestbench.WAIT.value} {KeywordsTestbench.FOR.value} 10*clk_period;"
        yield f'{KeywordsTestbench.REPORT.value} "======Tests finished======" severity Note;'
        yield f'{KeywordsTestbench.REPORT.value} "Please check the output message." severity Note;'
        yield f"{KeywordsTestbench.WAIT.value};"


class Procedure:
    def __init__(self, identifier: str):
        self.identifier = identifier
        self._declaration_list = InterfaceList()
        self._declaration_list_with_is = InterfaceList()
        self._statement_list = InterfaceList()

    @property
    def declaration_list(self):
        return self._declaration_list

    @declaration_list.setter
    def declaration_list(self, value):
        self._declaration_list = InterfaceList(value)

    @property
    def declaration_list_with_is(self):
        return self._declaration_list_with_is

    @declaration_list_with_is.setter
    def declaration_list_with_is(self, value):
        self._declaration_list_with_is = InterfaceList(value)

    @property
    def statement_list(self):
        return self._statement_list

    @statement_list.setter
    def statement_list(self, value):
        self._statement_list = InterfaceList(value)

    def __call__(self):
        yield f"{KeywordsTestbench.PROCEDURE.value} {self.identifier} ("
        if len(self._declaration_list) > 0:
            yield from _filter_empty_lines(
                _add_semicolons(self._declaration_list(), semicolon_last=True)
            )
        if len(self._declaration_list_with_is) > 0:
            yield from _filter_empty_lines(_add_is(self._declaration_list_with_is()))
        yield f"{Keywords.BEGIN.value}"
        if len(self._statement_list) > 0:
            yield from _filter_empty_lines(
                _add_semicolons(self._statement_list(), semicolon_last=True)
            )
        yield f"{Keywords.END.value} {self.identifier};"
