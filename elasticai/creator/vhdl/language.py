"""
The module contains classes and functions for generating vhdl code.
We provide code generators for the subset of vhdl that we need for implementing
our neural network accelerators and test benches. We stick closely to the vhdl
formal grammar with our class names.

The core of this module is the `CodeGenerator`. Code generators are callables that return `Code`.
`Code` is an iterable of strings. Depending on complexity we define syntactic components of the vhdl
grammar as `CodeGenerator`s. The class can then be used to set up and configure a function that yields lines
of code as strings.
"""
from collections import Sequence
from enum import Enum
from itertools import chain, filterfalse
from typing import Callable, Iterable, Literal, Optional, Union

Identifier = str
Code = Iterable[str]
CodeGenerator = Callable[[], Code]
CodeGeneratorCompatible = Union[Code, CodeGenerator, str]


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
        self, identifier: str, design_type: Literal[Keywords.ENTITY, Keywords.PORT]
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

    def _header(self) -> Code:
        if len(self.generic_list) > 0:
            yield from _clause(Keywords.GENERIC, self._generic_list())
        if len(self.port_list) > 0:
            yield from _clause(Keywords.PORT, self._port_list())

    def __call__(self) -> Code:
        yield f"{self.type.value} {self.identifier} {Keywords.IS.value}"
        yield from _filter_empty_lines(self._header())
        yield f"{Keywords.END.value} {self.type.value} {self.identifier};"


class Entity(_DesignUnitForEntityAndComponent):
    def __init__(self, identifier: str):
        super().__init__(identifier, Keywords.ENTITY)


class ComponentDeclaration(_DesignUnitForEntityAndComponent):
    def __init__(self, identifier: str):
        super().__init__(identifier, Keywords.COMPONENT)


class Architecture:
    def __init__(self, design_unit: str):
        self.design_unit = design_unit
        self._architecture_declaration_list = InterfaceList()
        self._architecture_component_list = InterfaceList()
        self._architecture_assignment_list = InterfaceList()
        self._architecture_process_list = InterfaceList()
        self._architecture_port_map_list = InterfaceList()
        self._architecture_assignment_at_end_of_declaration_list = InterfaceList()
        self._architecture_statement_part = InterfaceList()

    @property
    def architecture_declaration_list(self):
        return self._architecture_declaration_list

    @architecture_declaration_list.setter
    def architecture_declaration_list(self, value):
        self._architecture_declaration_list = InterfaceList(value)

    @property
    def architecture_statement_part(self):
        return self._architecture_statement_part

    @architecture_statement_part.setter
    def architecture_statement_part(self, value):
        self._architecture_statement_part = value

    @property
    def architecture_component_list(self):
        return self._architecture_component_list

    @architecture_component_list.setter
    def architecture_component_list(self, value):
        self._architecture_component_list = InterfaceList(value)

    @property
    def architecture_assignment_list(self):
        return self._architecture_assignment_list

    @architecture_assignment_list.setter
    def architecture_assignment_list(self, value):
        self._architecture_assignment_list = InterfaceList(value)

    @property
    def architecture_process_list(self):
        return self._architecture_process_list

    @architecture_process_list.setter
    def architecture_process_list(self, value):
        self._architecture_process_list = InterfaceList(value)

    @property
    def architecture_port_map_list(self):
        return self._architecture_port_map_list

    @architecture_port_map_list.setter
    def architecture_port_map_list(self, value):
        self._architecture_port_map_list = InterfaceList(value)

    @property
    def architecture_assignment_at_end_of_declaration_list(self):
        return self._architecture_assignment_at_end_of_declaration_list

    @architecture_assignment_at_end_of_declaration_list.setter
    def architecture_assignment_at_end_of_declaration_list(self, value):
        self._architecture_assignment_at_end_of_declaration_list = InterfaceList(value)

    def __call__(self) -> Code:
        yield f"{Keywords.ARCHITECTURE.value} rtl {Keywords.OF.value} {self.design_unit} {Keywords.IS.value}"
        if len(self._architecture_declaration_list) > 0:
            yield from _filter_empty_lines(
                _add_semicolons(
                    self._architecture_declaration_list(), semicolon_last=True
                )
            )
        if len(self._architecture_component_list) > 0:
            yield from _filter_empty_lines(self._architecture_component_list())
        yield f"{Keywords.BEGIN.value}"
        if len(self._architecture_assignment_list) > 0:
            yield from _filter_empty_lines(
                _add_semicolons(
                    self._architecture_assignment_list(), semicolon_last=True
                )
            )
        if len(self._architecture_process_list) > 0:
            yield from _filter_empty_lines(self.architecture_process_list())
        if len(self._architecture_port_map_list) > 0:
            yield from _filter_empty_lines(self.architecture_port_map_list())
        if len(self._architecture_assignment_at_end_of_declaration_list) > 0:
            yield from _filter_empty_lines(
                _add_semicolons(
                    self._architecture_assignment_at_end_of_declaration_list(),
                    semicolon_last=True,
                )
            )

        if self._architecture_statement_part:
            yield from _filter_empty_lines(self._architecture_statement_part())
        yield f"{Keywords.END.value} {Keywords.ARCHITECTURE.value} rtl;"


class Process:
    def __init__(
        self,
        identifier: str,
        input_name: str = None,
        lookup_table_generator_function: CodeGenerator = None,
    ):
        self.identifier = identifier
        self.process_declaration_list = []
        self.process_statements_list = []
        self.lookup_table_generator_function = lookup_table_generator_function
        self.input = input_name

    def _header(self) -> Code:
        if len(self.process_declaration_list) > 0:
            yield from _append_semicolons_to_lines(self.process_declaration_list)

    def _body(self) -> Code:
        if len(self.process_statements_list) > 0:
            yield from _append_semicolons_to_lines(self.process_statements_list)
        if self.lookup_table_generator_function:
            yield from self.lookup_table_generator_function

    def __call__(self) -> Code:
        if self.input:
            yield f"{self.identifier}_{Keywords.PROCESS.value}: {Keywords.PROCESS.value}({self.input})"
        else:
            yield f"{self.identifier}_{Keywords.PROCESS.value}: {Keywords.PROCESS.value}"
        yield from _filter_empty_lines(self._header())
        yield f"{Keywords.BEGIN.value}"
        yield from _filter_empty_lines(self._body())
        yield f"{Keywords.END.value} {Keywords.PROCESS.value} {self.identifier}_{Keywords.PROCESS.value};"


class ContextClause:
    def __init__(self, library_clause, use_clause):
        self._use_clause = use_clause
        self._library_clause = library_clause

    def __call__(self):
        yield from self._library_clause()
        yield from self._use_clause()


class UseClause:
    def __init__(self, selected_names: list[str]):
        self._selected_names = selected_names

    def __call__(self):
        def prefix_use(line: str):
            return f"use {line}"

        yield from _append_semicolons_to_lines(map(prefix_use, self._selected_names))


class LibraryClause:
    def __init__(self, logical_name_list: list[str]):
        self._logical_name_list = logical_name_list

    def __call__(self):
        yield from _append_semicolons_to_lines(
            ["library {}".format(", ".join(self._logical_name_list))]
        )


class InterfaceConstrained:
    def __init__(
        self,
        identifier: str,
        identifier_type: DataType,
        range: Optional[Union[str, int]],
        mode: Optional[Mode],
        value: Optional[Union[str, int]],
        declaration_type: Optional[str],
    ):
        self._identifier = identifier
        self._range = f"({range})" if range else ""
        self._identifier_type = identifier_type
        self._mode = f" {mode.value} " if mode else " "
        self._value = f" := {value}" if value else ""
        self._declaration_type = f"{declaration_type} " if declaration_type else ""

    def __call__(self) -> Code:
        yield from (
            f"{self._declaration_type}{self._identifier} :{self._mode}{self._identifier_type.value}{self._range}{self._value}",
        )


class InterfaceSignal(InterfaceConstrained):
    def __init__(
        self,
        identifier: str,
        identifier_type: DataType,
        range: Optional[Union[str, int]] = None,
        mode: Optional[Mode] = None,
        value: Optional[Union[str, int]] = None,
    ):
        super().__init__(
            identifier, identifier_type, range, mode, value, declaration_type="signal"
        )


class InterfaceVariable(InterfaceConstrained):
    def __init__(
        self,
        identifier: str,
        identifier_type: DataType,
        range: Optional[Union[str, int]] = None,
        mode: Optional[Mode] = None,
        value: Optional[Union[str, int]] = None,
    ):
        super().__init__(
            identifier, identifier_type, range, mode, value, declaration_type=None
        )


class CodeGeneratorConcatenation(Sequence[CodeGenerator]):
    def __len__(self) -> int:
        return len(self.interface_generators)

    def __getitem__(self, item) -> CodeGenerator:
        return self.interface_generators.__getitem__(item)

    def __init__(self, *interfaces: CodeGeneratorCompatible):
        self.interface_generators = [
            _unify_code_generators(interface) for interface in interfaces
        ]

    def append(self, interface: CodeGeneratorCompatible) -> None:
        self.interface_generators.append(_unify_code_generators(interface))

    def __call__(self) -> Code:
        yield from chain.from_iterable(
            (interface() for interface in self.interface_generators)
        )


class PortMap:
    def __init__(self, map_name, component_name):
        self.map_name = map_name
        self.component_name = component_name
        self._signal_list = InterfaceList()
        self._generic_map_list = InterfaceList()

    @property
    def signal_list(self):
        return self._signal_list

    @signal_list.setter
    def signal_list(self, value):
        self._signal_list = InterfaceList(value)

    @property
    def generic_map_list(self):
        return self._generic_map_list

    @generic_map_list.setter
    def generic_map_list(self, value):
        self._generic_map_list = InterfaceList(value)

    def __call__(self) -> Code:
        yield f"{self.map_name}: {self.component_name}"
        if len(self._generic_map_list) > 0:
            yield f"generic map ("
            yield from _filter_empty_lines(_add_comma(self._generic_map_list()))
            yield ")"
        yield f"port map ("
        yield from _filter_empty_lines(_add_comma(self._signal_list()))
        yield ");"


class InterfaceList(CodeGeneratorConcatenation):
    pass


InterfaceDeclaration = Union[
    "InterfaceObjectDeclaration",
    "InterfaceTypeDeclaration",
    "InterfaceSubprogramDeclaration",
    "InterfacePackageDeclaration",
]

InterfaceObjectDeclaration = Union[
    "InterfaceConstantDeclaration",
    "InterfaceSignalDeclaration",
    "InterfaceVariableDeclaration",
    "InterfaceFileDeclaration",
]

ClauseType = Literal[Keywords.GENERIC, Keywords.PORT]


def indent(line: str) -> str:
    return "".join(["\t", line])


def _add_semicolons(lines: Code, semicolon_last: bool = False) -> Code:
    temp = tuple(lines)
    yield from (f"{line};" for line in temp[:-1])
    yield f"{temp[-1]};" if semicolon_last else f"{temp[-1]}"


def _add_is(lines: Code) -> Code:
    yield from (f"{line} is" for line in lines)


def _add_comma(lines: Code, comma_last: bool = False) -> Code:
    temp = tuple(lines)
    yield from (f"{line}," for line in temp[:-1])
    yield f"{temp[-1]}," if comma_last else f"{temp[-1]}"


def _append_semicolons_to_lines(lines: Code) -> Code:
    temp = tuple(lines)
    yield from (f"{line};" for line in temp)


def _clause(clause_type: ClauseType, interfaces: Code) -> Code:
    yield f"{clause_type.value} ("
    yield from _filter_empty_lines(_add_semicolons(interfaces))
    yield ");"


def _filter_empty_lines(lines: Code) -> Code:
    return filterfalse(_line_is_empty, lines)


def _line_is_empty(line: str) -> bool:
    return len(line) == 0


def _join_lines(lines) -> str:
    return "\n".join(lines)


def _empty_code_generator() -> Code:
    return []


def form_to_hex_list(lines: Code):
    hex_string = "".join(f'x"{line}",' for line in lines[:-1])
    return hex_string + f'x"{lines[-1]}"'


# def _indent_and_filter_non_empty_lines(lines: Code) -> Code:
#     yield from map(indent, _filter_empty_lines(lines))


# noinspection PyPep8Naming
def _wrap_in_IS_END_block(
    block_type: Keywords, block_identifier: Identifier, lines: Code
) -> Code:
    yield f"{block_type.value} {block_identifier} {Keywords.IS.value}"
    yield from _filter_empty_lines(lines)
    yield f"{Keywords.END.value} {block_type.value} {block_identifier};"


def _wrap_string_into_code_generator(string: str) -> CodeGenerator:
    def wrapped():
        return (string,)

    return wrapped


def _wrap_code_into_code_generator(code: Code) -> CodeGenerator:
    def wrapped():
        return code

    return wrapped


def _unify_code_generators(generator: CodeGeneratorCompatible) -> CodeGenerator:
    if isinstance(generator, str):
        return _wrap_string_into_code_generator(generator)
    elif isinstance(generator, Iterable):
        return _wrap_code_into_code_generator(generator)
    elif isinstance(generator, Callable):
        return generator
    else:
        raise ValueError


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
        yield f"procedure {self.identifier} ("
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
