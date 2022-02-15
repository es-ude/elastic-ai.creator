import math
from abc import ABC, abstractmethod
from itertools import chain
from typing import Iterable, Union

import torch.nn

from elasticai.creator.vhdl.generator.generator_functions import (
    precomputed_scalar_function_process, precomputed_logic_function_process,
)
from elasticai.creator.vhdl.language import (
    Entity,
    InterfaceVariable,
    DataType,
    Code,
    _wrap_in_IS_END_block,
    indent,
    Architecture,
    Process,
    ContextClause,
    LibraryClause,
    UseClause, InterfaceConstrained, Mode,
)
from elasticai.creator.vhdl.number_representations import BitVector


class DataWidthVariable(InterfaceVariable):
    def __init__(self, value: int):
        super().__init__(
            identifier="DATA_WIDTH", variable_type=DataType.INTEGER, value=f"{value}"
        )


class FracWidthVariable(InterfaceVariable):
    def __init__(self, value: int):
        super().__init__(
            identifier="FRAC_WIDTH", variable_type=DataType.INTEGER, value=f"{value}"
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

    @property
    def architecture_name(self) -> str:
        return f"{self.component_name}_rtl"

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
            identifier=self.architecture_name,
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

    def build(self) -> str:
        lines_of_code = self.__call__()
        return "\n".join(lines_of_code)


class NaiveLUTBasedConv():
    def __init__(self, component_name: str, inputs: Iterable[list[BitVector]], outputs: Iterable[list[BitVector]], input_width:int, output_width:int
                 ):
        self.identifier = component_name
        self.inputs = inputs
        self.outputs = outputs
        self.input_width = input_width
        self.output_width = output_width

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
        entity = Entity("NaiveLUTConv")
        entity.port_list.append(InterfaceConstrained(
            identifier="x",
            mode=Mode.IN,
            range=f"INPUT_WIDTH-1 downto 0",
            variable_type=DataType.STD_LOGIC_VECTOR,
        ))
        entity.port_list.append(InterfaceConstrained(
            identifier="y",
            mode=Mode.OUT,
            range=f"OUTPUT_WIDTH-1 downto 0",
            variable_type=DataType.STD_LOGIC_VECTOR,
        ))
        entity.generic_list = [
            f"INPUT_WIDTH : integer := {self.input_width}",
            f"OUTPUT_WIDTH : integer := {self.output_width}",
        ]
        architecture = Architecture(
            identifier=self.identifier,
            design_unit="NaiveLUTConv",
        )
        architecture.architecture_statement_part = lambda :precomputed_logic_function_process(
                x_list=self.inputs, y_list=self.outputs
            )
        code = chain(chain(library(), entity()), architecture())
        return code
