import math
from abc import abstractmethod
from enum import Enum
from itertools import chain, filterfalse
from typing import (
    Callable,
    Iterable,
    Iterator,
    Literal,
    Optional,
    Protocol,
    Sequence,
    overload,
    runtime_checkable,
)

import torch.nn

from elasticai.creator.hdl.vhdl.code_generation.code_generation import (
    bin_representation,
)
from elasticai.creator.vhdl.number_representations import (
    FixedPoint,
    float_values_to_fixed_point,
    infer_total_and_frac_bits,
)

Code = Iterable[str]


@runtime_checkable
class CodeGenerator(Protocol):
    @abstractmethod
    def code(self) -> Iterable[str]:
        ...


CodeGeneratorCompatible = Code | CodeGenerator | str | Callable[[], Code]


def _vhdl_add_assignment(
    code: list[str], line_id: str, value: str, comment: Optional[str] = None
) -> None:
    new_code_fragment = f"{line_id} <= {bin_representation(value)};"
    if comment is not None:
        new_code_fragment += f" -- {comment}"
    code.append(new_code_fragment)


def _get_lower_case_class_name_or_component_name(
    cls: type, component_name: Optional[str]
) -> str:
    if component_name is None:
        return cls.__name__.lower()
    return component_name


def precomputed_scalar_function_process(
    x: list[FixedPoint], y: list[FixedPoint]
) -> list[str]:
    """
        returns the string of a lookup table
    Args:
        x (FixedPoint): input list
        y (FixedPoint) : output list
    Returns:
        String of lookup table (if/elsif statements for vhdl file)
    """
    x.sort()
    lines: list[str] = []
    if len(x) == 0 and len(y) == 1:
        _vhdl_add_assignment(
            code=lines,
            line_id="y",
            value=y[0].to_bin(),
        )
    elif len(x) != len(y) - 1:
        raise ValueError(
            f"x has to be one element shorter than y, but x has {len(x)} elements and y"
            f" {len(y)} elements"
        )
    else:
        smallest_possible_output = y[0]
        biggest_possible_output = y[-1]

        # first element
        for x_value in x[:1]:
            lines.append(f"if int_x<{x_value.to_signed_int()} then")
            _vhdl_add_assignment(
                code=lines,
                line_id="y",
                value=smallest_possible_output.to_bin(),
                comment=str(smallest_possible_output.to_signed_int()),
            )
            lines[-1] = "\t" + lines[-1]
        for current_x, current_y in zip(x[1:], y[1:-1]):
            lines.append(f"elsif int_x<{current_x.to_signed_int()} then")
            _vhdl_add_assignment(
                code=lines,
                line_id="y",
                value=current_y.to_bin(),
                comment=str(current_y.to_signed_int()),
            )
            lines[-1] = "\t" + lines[-1]
        # last element only in y
        for _ in y[-1:]:
            lines.append("else")
            _vhdl_add_assignment(
                code=lines,
                line_id="y",
                value=biggest_possible_output.to_bin(),
                comment=str(biggest_possible_output.to_signed_int()),
            )
            lines[-1] = "\t" + lines[-1]
        if len(lines) != 0:
            lines.append("end if;")

    # build the string block

    return lines


class PrecomputedScalarFunction:
    def __init__(
        self,
        x: list[FixedPoint],
        y: list[FixedPoint],
        component_name: Optional[str] = None,
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
        self.component_name = _get_lower_case_class_name_or_component_name(
            cls=type(self), component_name=component_name
        )

        self.data_width, self.frac_width = infer_total_and_frac_bits(x, y)
        self.x = x
        self.y = y
        self.process_instance = None

    @property
    def file_name(self) -> str:
        return f"{self.component_name}.vhd"

    def code(self) -> list[str]:
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
            lookup_table=precomputed_scalar_function_process(x=self.x, y=self.y),
            input_name="x",
        )
        process.process_declaration_list = ["variable int_x: integer := 0"]
        process.process_statements_list = ["int_x := to_integer(x)"]
        architecture = Architecture(
            design_unit=self.component_name,
        )
        architecture.architecture_statement_part = process
        code = list(chain(chain(library.code(), entity.code()), architecture.code()))
        return code


class Sigmoid(PrecomputedScalarFunction):
    def __init__(
        self, x: list[FixedPoint], component_name: Optional[str] = None
    ) -> None:
        x_tensor = torch.as_tensor(list(map(float, x)))
        # calculate y always for the previous element therefore, the last input is not needed here
        y = torch.nn.Sigmoid()(x_tensor[:-1]).tolist()
        y.insert(0, 0)
        # add last y value, therefore, x_tensor is one element shorter than y_tensor
        y.append(1)
        y = float_values_to_fixed_point(y, *infer_total_and_frac_bits(x))

        super(Sigmoid, self).__init__(x=x, y=y, component_name=component_name)


class Tanh(PrecomputedScalarFunction):
    def __init__(
        self, x: list[FixedPoint], component_name: Optional[str] = None
    ) -> None:
        y_list = [-1.0]
        # calculate y always for the previous element, therefore, the last input is not needed here
        for x_element in x[:-1]:
            y_list.append(math.tanh(float(x_element)))
        # add last y value, therefore, x_list is one element shorter than y_list
        y_list.append(1)
        y = float_values_to_fixed_point(y_list, *infer_total_and_frac_bits(x))

        super(Tanh, self).__init__(x=x, y=y, component_name=component_name)


Identifier = str


class Keywords(Enum):
    IS = "is"
    END = "end"
    ENTITY = "entity"
    COMPONENT = "component"
    PORT = "port"
    GENERIC = "generic"
    IN = "in"
    OUT = "out"
    INOUT = "inout"
    BUFFER = "buffer"
    LINKAGE = "linkage"
    INTEGER = "integer"
    STD_LOGIC = "std_logic"
    SIGNAL = "signal"
    ARCHITECTURE = "architecture"
    OF = "of"
    SIGNED = "signed"
    BEGIN = "begin"
    PROCESS = "process"
    STD_LOGIC_VECTOR = "std_logic_vector"


class DataType(Enum):
    INTEGER = Keywords.INTEGER.value
    STD_LOGIC = Keywords.STD_LOGIC.value
    SIGNED = Keywords.SIGNED.value
    STD_LOGIC_VECTOR = Keywords.STD_LOGIC_VECTOR.value


class Mode(Enum):
    IN = Keywords.IN.value
    OUT = Keywords.OUT.value
    INOUT = Keywords.INOUT.value
    BUFFER = Keywords.BUFFER.value


class _DesignUnitForEntityAndComponent:
    def __init__(
        self, identifier: str, design_type: Literal[Keywords.ENTITY, Keywords.COMPONENT]
    ):
        self.identifier = identifier
        self._generic_list = InterfaceList()
        self._port_list = InterfaceList()
        self.type = design_type

    @property
    def generic_list(self):
        return self._generic_list

    @generic_list.setter
    def generic_list(self, value):
        self._generic_list = InterfaceList(value)

    @property
    def port_list(self):
        return self._port_list

    @port_list.setter
    def port_list(self, value):
        self._port_list = InterfaceList(value)

    def _header(self) -> Iterator[str]:
        if len(self.generic_list) > 0:
            yield from _clause(Keywords.GENERIC, self._generic_list.code())
        if len(self.port_list) > 0:
            yield from _clause(Keywords.PORT, self._port_list.code())

    def __iter__(self) -> Iterator[str]:
        yield f"{self.type.value} {self.identifier} {Keywords.IS.value}"
        yield from _filter_empty_lines(self._header())
        yield f"{Keywords.END.value} {self.type.value} {self.identifier};"

    def code(self) -> Iterable[str]:
        return self


class Entity(_DesignUnitForEntityAndComponent):
    def __init__(self, identifier: str):
        super().__init__(identifier, Keywords.ENTITY)


class Architecture:
    def __init__(self, design_unit: str):
        self.design_unit = design_unit
        self.architecture_declaration_list = InterfaceList()
        self.architecture_component_list = InterfaceList()
        self.architecture_assignment_list = InterfaceList()
        self.architecture_process_list = InterfaceList()
        self.architecture_port_map_list = InterfaceList()
        self.architecture_assignment_at_end_of_declaration_list = InterfaceList()
        self.architecture_statement_part: Optional[CodeGenerator] = None

    def _generator(self) -> Iterator[str]:
        yield (
            f"{Keywords.ARCHITECTURE.value} rtl"
            f" {Keywords.OF.value} {self.design_unit} {Keywords.IS.value}"
        )
        if len(self.architecture_declaration_list) > 0:
            yield from _filter_empty_lines(
                _add_semicolons(
                    self.architecture_declaration_list.code(), semicolon_last=True
                )
            )
        if len(self.architecture_component_list) > 0:
            yield from _filter_empty_lines(self.architecture_component_list.code())
        yield f"{Keywords.BEGIN.value}"
        if len(self.architecture_assignment_list) > 0:
            yield from _filter_empty_lines(
                _add_semicolons(
                    self.architecture_assignment_list.code(), semicolon_last=True
                )
            )
        if len(self.architecture_process_list) > 0:
            yield from _filter_empty_lines(self.architecture_process_list.code())
        if len(self.architecture_port_map_list) > 0:
            yield from _filter_empty_lines(self.architecture_port_map_list.code())
        if len(self.architecture_assignment_at_end_of_declaration_list) > 0:
            yield from _filter_empty_lines(
                _add_semicolons(
                    self.architecture_assignment_at_end_of_declaration_list.code(),
                    semicolon_last=True,
                )
            )

        if self.architecture_statement_part:
            yield from _filter_empty_lines(self.architecture_statement_part.code())
        yield f"{Keywords.END.value} {Keywords.ARCHITECTURE.value} rtl;"

    def code(self) -> Code:
        return self._generator()


class Process(Code, CodeGenerator):
    def __init__(
        self,
        identifier: str,
        input_name: Optional[str] = None,
        lookup_table: Optional[Code] = None,
    ):
        self.identifier = identifier
        self.process_declaration_list: list[str] = []
        self.process_statements_list: list[str] = []
        self.lookup_table = lookup_table
        self.input = input_name

    def _header(self) -> Iterator[str]:
        if len(self.process_declaration_list) > 0:
            yield from _append_semicolons_to_lines(self.process_declaration_list)

    def _body(self) -> Iterator[str]:
        if len(self.process_statements_list) > 0:
            yield from _append_semicolons_to_lines(self.process_statements_list)
        if self.lookup_table:
            yield from self.lookup_table

    def __iter__(self) -> Iterator[str]:
        if self.input:
            yield (
                f"{self.identifier}_{Keywords.PROCESS.value}:"
                f" {Keywords.PROCESS.value}({self.input})"
            )
        else:
            yield (
                f"{self.identifier}_{Keywords.PROCESS.value}: {Keywords.PROCESS.value}"
            )
        yield from _filter_empty_lines(self._header())
        yield f"{Keywords.BEGIN.value}"
        yield from _filter_empty_lines(self._body())
        yield (
            f"{Keywords.END.value} {Keywords.PROCESS.value} {self.identifier}_{Keywords.PROCESS.value};"
        )

    def code(self):
        return self


class ContextClause(Code, CodeGenerator):
    def __init__(self, library_clause, use_clause):
        self._use_clause = use_clause
        self._library_clause = library_clause

    def __iter__(self) -> Iterator[str]:
        yield from self._library_clause
        yield from self._use_clause

    def code(self) -> Code:
        return self


class UseClause(Code, CodeGenerator):
    def __init__(self, selected_names: list[str]):
        self._selected_names = selected_names

    def __iter__(self):
        def prefix_use(line: str):
            return f"use {line}"

        yield from _append_semicolons_to_lines(map(prefix_use, self._selected_names))

    def code(self) -> Code:
        return self


class LibraryClause(Code, CodeGenerator):
    def __init__(self, logical_name_list: list[str]):
        self._logical_name_list = logical_name_list

    def __iter__(self):
        yield from _append_semicolons_to_lines(
            ["library {}".format(", ".join(self._logical_name_list))]
        )

    def code(self) -> Code:
        return self


def _wrap_iterator_function_into_iterable(create_iterator: Callable[[], Iterator[str]]):
    class Wrapper(Iterable[str]):
        def __iter__(self) -> Iterator[str]:
            yield from create_iterator()

    wrapped = Wrapper()
    return wrapped


class CodeGeneratorConcatenation(Sequence[CodeGenerator], CodeGenerator):
    def __init__(self, *interfaces: CodeGeneratorCompatible):
        self.interface_generators: list[CodeGenerator] = [
            _unify_code_generators(interface) for interface in interfaces
        ]

    @overload
    def __getitem__(self, index: int) -> CodeGenerator:
        ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[CodeGenerator]:
        ...

    def __getitem__(self, index):
        if isinstance(index, slice):
            return CodeGeneratorConcatenation(*tuple(self.interface_generators[index]))
        else:
            return self.interface_generators[index]

    def __len__(self) -> int:
        return len(self.interface_generators)

    def append(self, interface: CodeGeneratorCompatible) -> None:
        self.interface_generators.append(_unify_code_generators(interface))

    def _code_iterator(self) -> Iterator[str]:
        for generator in self.interface_generators:
            for line in generator.code():
                yield line

    def code(self) -> Code:
        return _wrap_iterator_function_into_iterable(self._code_iterator)


class InterfaceList(CodeGeneratorConcatenation):
    pass


ClauseType = Literal[Keywords.GENERIC, Keywords.COMPONENT, Keywords.PORT]


def _add_semicolons(lines: Code, semicolon_last: bool = False) -> Iterator[str]:
    temp = tuple(lines)
    yield from (f"{line};" for line in temp[:-1])
    yield f"{temp[-1]};" if semicolon_last else f"{temp[-1]}"


def _add_is(lines: Code) -> Iterator[str]:
    yield from (f"{line} is" for line in lines)


def _add_comma(lines: Code, comma_last: bool = False) -> Iterator[str]:
    temp = tuple(lines)
    yield from (f"{line}," for line in temp[:-1])
    yield f"{temp[-1]}," if comma_last else f"{temp[-1]}"


def _append_semicolons_to_lines(lines: Code) -> Iterator[str]:
    temp = tuple(lines)
    yield from (f"{line};" for line in temp)


def _clause(clause_type: ClauseType, interfaces: Code) -> Iterator[str]:
    yield f"{clause_type.value} ("
    yield from _filter_empty_lines(_add_semicolons(interfaces))
    yield ");"


def _filter_empty_lines(lines: Code) -> Iterator[str]:
    return filterfalse(_line_is_empty, lines)


def _line_is_empty(line: str) -> bool:
    return len(line) == 0


def _join_lines(lines) -> str:
    return "\n".join(lines)


def _empty_code_generator() -> Code:
    return []


def _wrap_in_is_end_block(
    block_type: Keywords, block_identifier: Identifier, lines: Code
) -> Iterator[str]:
    yield f"{block_type.value} {block_identifier} {Keywords.IS.value}"
    yield from _filter_empty_lines(lines)
    yield f"{Keywords.END.value} {block_type.value} {block_identifier};"


def _wrap_string_into_code_generator(string: str) -> CodeGenerator:
    class Wrapper(CodeGenerator):
        def code(self) -> Code:
            return [string]

    return Wrapper()


def _wrap_code_into_code_generator(code: Code) -> CodeGenerator:
    class Wrapped(CodeGenerator):
        def code(self) -> Code:
            return code

    wrapped = Wrapped()
    return wrapped


def _wrap_callable_into_code_generator(fn: Callable[[], Code]) -> CodeGenerator:
    class Wrapper(CodeGenerator):
        def code(self) -> Code:
            return fn()

    wrapped = Wrapper()
    return wrapped


def _unify_code_generators(generator: CodeGeneratorCompatible) -> CodeGenerator:
    if isinstance(generator, str):
        return _wrap_string_into_code_generator(generator)
    elif isinstance(generator, Iterable):
        return _wrap_code_into_code_generator(generator)
    elif isinstance(generator, CodeGenerator):
        return generator
    elif callable(generator):
        return _wrap_callable_into_code_generator(generator)
    else:
        raise ValueError
