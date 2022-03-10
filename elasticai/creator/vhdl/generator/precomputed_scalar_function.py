import math
from itertools import chain
from typing import Iterable

import torch.nn

from elasticai.creator.vhdl.generator.generator_functions import (
    precomputed_scalar_function_process,
)
from elasticai.creator.vhdl.language import (
    Entity,
    ComponentDeclaration,
    InterfaceVariable,
    DataType,
    Architecture,
    Process,
    ContextClause,
    LibraryClause,
    UseClause,
    PortMap,
)
from elasticai.creator.vhdl.language_testbench import TestCases


class DataWidthVariable(InterfaceVariable):
    def __init__(self, value: int):
        super().__init__(
            identifier="DATA_WIDTH", identifier_type=DataType.INTEGER, value=f"{value}"
        )


class FracWidthVariable(InterfaceVariable):
    def __init__(self, value: int):
        super().__init__(
            identifier="FRAC_WIDTH", identifier_type=DataType.INTEGER, value=f"{value}"
        )


class PrecomputedScalarFunction:
    def __init__(
        self, data_width, frac_width, x, y, component_name=None, process_instance=None
    ):
        """
        We calculate the function with an algorithm equivalent to:
        ```
        def function(x: int, inputs: list[int], outputs: list[int]) -> int:
          for input, output in zip(inputs, outputs[:-1]):
            if x < input:
              return output
          return outputs[-1]
        ```
        """
        self.component_name = self._get_lower_case_class_name_or_component_name(
            component_name=component_name
        )
        self.data_width = data_width
        self.frac_width = frac_width
        self.x = x
        self.y = y
        self.process_instance = process_instance

    @classmethod
    def _get_lower_case_class_name_or_component_name(cls, component_name):
        if component_name is None:
            return cls.__name__.lower()
        return component_name

    @property
    def file_name(self) -> str:
        return f"{self.component_name}.vhd"

    def __call__(self) -> Iterable[str]:
        library = ContextClause(
            library_clause=LibraryClause(logical_name_list=["ieee"]),
            use_clause=UseClause(
                selected_names=[
                    "ieee.std_logic_1164.all",
                    "ieee.numeric_std.all",
                ]
            ),
        )
        entity = Entity(self.component_name)
        entity.generic_list = [
            f"DATA_WIDTH : integer := {self.data_width}",
            f"FRAC_WIDTH : integer := {self.frac_width}",
        ]
        entity.port_list = [
            "x : in signed(DATA_WIDTH-1 downto 0)",
            "y : out signed(DATA_WIDTH-1 downto 0)",
        ]
        process = Process(
            identifier=self.component_name,
            lookup_table_generator_function=precomputed_scalar_function_process(
                x_list=self.x, y_list=self.y
            ),
            input_name="x",
        )
        process.process_declaration_list = ["variable int_x: integer := 0"]
        process.process_statements_list = ["int_x := to_integer(x)"]
        architecture = Architecture(
            design_unit=self.component_name,
        )
        architecture.architecture_statement_part = process
        code = chain(chain(library(), entity()), architecture())
        return code


class Sigmoid(PrecomputedScalarFunction):
    def __init__(self, data_width, frac_width, x, component_name=None):
        x_list = torch.as_tensor(x)
        # calculate y always for the previous element, therefore the last input is not needed here
        y_list = list(torch.nn.Sigmoid()(x_list[:-1]))
        y_list.insert(0, 0)
        # add last y value, therefore, x_list is one element shorter than y_list
        y_list.append(1)
        super(Sigmoid, self).__init__(
            data_width=data_width,
            frac_width=frac_width,
            x=x,
            y=y_list,
            component_name=component_name,
        )


class Tanh(PrecomputedScalarFunction):
    def __init__(self, data_width, frac_width, x, component_name=None):
        y_list = [-1]
        # calculate y always for the previous element, therefore the last input is not needed here
        for x_element in x[:-1]:
            y_list.append(math.tanh(x_element))
        # add last y value, therefore, x_list is one element shorter than y_list
        y_list.append(1)
        super(Tanh, self).__init__(
            data_width=data_width,
            frac_width=frac_width,
            x=x,
            y=y_list,
            component_name=component_name,
        )


class PrecomputedScalarTestBench:
    def __init__(
        self,
        data_width: int,
        frac_width: int,
        x_list_for_testing: list,
        y_list_for_testing: list,
        component_name: str = None,
    ):
        self.component_name = self._get_lower_case_class_name_or_component_name(
            component_name=component_name
        )
        self.data_width = data_width
        self.frac_width = frac_width
        self.x_list_for_testing = x_list_for_testing
        self.y_list_for_testing = y_list_for_testing

    @classmethod
    def _get_lower_case_class_name_or_component_name(cls, component_name):
        if component_name is None:
            return cls.__name__.lower()
        return component_name

    @property
    def file_name(self) -> str:
        return f"{self.component_name}_tb.vhd"

    def __call__(self) -> Iterable[str]:
        library = ContextClause(
            library_clause=LibraryClause(logical_name_list=["ieee"]),
            use_clause=UseClause(
                selected_names=[
                    "ieee.std_logic_1164.all",
                    "ieee.numeric_std.all",
                    "ieee.math_real.all",
                ]
            ),
        )

        entity = Entity(self.component_name + "_tb")
        entity.generic_list = [
            f"DATA_WIDTH : integer := {self.data_width}",
            f"FRAC_WIDTH : integer := {self.frac_width}",
        ]
        entity.port_list = [
            "clk : out std_logic",
        ]

        component = ComponentDeclaration(identifier=self.component_name)
        component.generic_list = [
            f"DATA_WIDTH : integer := {self.data_width}",
            f"FRAC_WIDTH : integer := {self.frac_width}",
        ]
        component.port_list = [
            "x : in signed(DATA_WIDTH-1 downto 0)",
            "y : out signed(DATA_WIDTH-1 downto 0)",
        ]

        process = Process(
            identifier="clock",
        )
        process.process_statements_list = [
            "clk <= '0'",
            "wait for clk_period/2",
            "clk <= '1'",
            "wait for clk_period/2",
        ]

        uut_port_map = PortMap(map_name="uut", component_name=self.component_name)
        uut_port_map.signal_list.append("x => test_input")
        uut_port_map.signal_list.append("y => test_output")

        test_cases = TestCases(
            x_list_for_testing=self.x_list_for_testing,
            y_list_for_testing=self.y_list_for_testing,
            data_width=self.data_width,
        )
        test_process = Process(identifier="test")
        test_process.process_test_case_list = test_cases

        architecture = Architecture(
            design_unit=self.component_name + "_tb",
        )
        architecture.architecture_declaration_list = [
            "signal clk_period : time := 1 ns",
            "signal test_input : signed(16-1 downto 0):=(others=>'0')",
            "signal test_output : signed(16-1 downto 0)",
        ]
        # FIXME: doesn't work like this
        # architecture.architecture_component_list = [component]
        architecture.architecture_component_list.append(component)
        architecture.architecture_process_list.append(process)
        architecture.architecture_port_map_list.append(uut_port_map)
        architecture.architecture_statement_part = test_process

        code = chain(chain(library(), entity()), architecture())
        return code
