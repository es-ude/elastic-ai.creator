import math
from itertools import chain
from typing import Iterable, Iterator

import torch.nn

from elasticai.creator.vhdl.language import (
    Architecture,
    Code,
    CodeGenerator,
    ComponentDeclaration,
    ContextClause,
    DataType,
    Entity,
    InterfaceVariable,
    LibraryClause,
    PortMap,
    Process,
    UseClause,
)
from elasticai.creator.vhdl.language_testbench import TestBenchBase
from elasticai.creator.vhdl.number_representations import (
    FloatToBinaryFixedPointStringConverter,
    FloatToSignedFixedPointConverter,
)


def _vhdl_add_assignment(code: list, line_id: str, value: str, comment=None) -> None:
    new_code_fragment = f'{line_id} <= "{value}";'
    if comment is not None:
        new_code_fragment += f" -- {comment}"
    code.append(new_code_fragment)


def precomputed_scalar_function_process(x_list, y_list) -> CodeGenerator:
    """
        returns the string of a lookup table
    Args:
        y_list : output List contains integers
        x_list: input List contains integers
    Returns:
        String of lookup table (if/elsif statements for vhdl file)
    """
    as_signed_fixed_point = FloatToSignedFixedPointConverter(
        bits_used_for_fraction=8, strict=False
    )
    as_binary_string = FloatToBinaryFixedPointStringConverter(
        total_bit_width=16, as_signed_fixed_point=as_signed_fixed_point
    )
    x_list.sort()
    lines = []
    if len(x_list) == 0 and len(y_list) == 1:
        _vhdl_add_assignment(
            code=lines,
            line_id="y",
            value=as_binary_string(y_list[0]),
        )
    elif len(x_list) != len(y_list) - 1:
        raise ValueError(
            "x_list has to be one element shorter than y_list, but x_list has {} elements and y_list {} elements".format(
                len(x_list), len(y_list)
            )
        )
    else:
        smallest_possible_output = y_list[0]
        biggest_possible_output = y_list[-1]

        # first element
        for x in x_list[:1]:
            lines.append("if int_x<{0} then".format(as_signed_fixed_point(x)))
            _vhdl_add_assignment(
                code=lines,
                line_id="y",
                value=as_binary_string(smallest_possible_output),
                comment=as_signed_fixed_point(smallest_possible_output),
            )
            lines[-1] = "\t" + lines[-1]
        for current_x, current_y in zip(x_list[1:], y_list[1:-1]):
            lines.append(
                "elsif int_x<{0} then".format(as_signed_fixed_point(current_x))
            )
            _vhdl_add_assignment(
                code=lines,
                line_id="y",
                value=as_binary_string(current_y),
                comment=as_signed_fixed_point(current_y),
            )
            lines[-1] = "\t" + lines[-1]
        # last element only in y
        for _ in y_list[-1:]:
            lines.append("else")
            _vhdl_add_assignment(
                code=lines,
                line_id="y",
                value=as_binary_string(biggest_possible_output),
                comment=as_signed_fixed_point(biggest_possible_output),
            )
            lines[-1] = "\t" + lines[-1]
        if len(lines) != 0:
            lines.append("end if;")
    # build the string block
    yield lines[0]
    for line in lines[1:]:
        yield line


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

        test_cases = TestCasesPrecomputedScalarFunction(
            x_list_for_testing=self.x_list_for_testing,
            y_list_for_testing=self.y_list_for_testing,
            data_width=self.data_width,
        )
        test_process = Process(identifier="test")
        test_process.process_statements_list = [t for t in test_cases()]

        architecture = Architecture(
            design_unit=self.component_name + "_tb",
        )
        architecture.architecture_declaration_list = [
            "signal clk_period : time := 1 ns",
            "signal test_input : signed(16-1 downto 0):=(others=>'0')",
            "signal test_output : signed(16-1 downto 0)",
        ]

        architecture.architecture_component_list.append(component)
        architecture.architecture_process_list.append(process)
        architecture.architecture_port_map_list.append(uut_port_map)
        architecture.architecture_statement_part = test_process

        code = chain(library(), entity(), architecture())
        return code


class TestCasesPrecomputedScalarFunction(TestBenchBase):
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

    def __len__(self):
        return len(self.y_list_for_testing)

    def _body(self) -> Iterator[str]:
        for x_value, y_value in zip(self.x_list_for_testing, self.y_list_for_testing):
            yield f"{self.x_variable_name} <= to_signed({x_value},{self.data_width})"
            yield f"wait for 1*clk_period"
            yield f"report \"The value of '{self.y_variable_name}' is \" & integer'image(to_integer(unsigned({self.y_variable_name})))"
            if isinstance(y_value, str):
                yield f'assert {self.y_variable_name}="{y_value}" report "The test case {x_value} fail" severity failure'
            else:
                yield f'assert {self.y_variable_name}={y_value} report "The test case {x_value} fail" severity failure'

    def __call__(self) -> Iterable[Code]:
        yield from iter(self)
