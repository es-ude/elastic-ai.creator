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
from itertools import filterfalse, chain
from typing import (
    Callable,
    Iterable,
    Union,
    Literal,
    Optional,
)

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


class DataType(Enum):
    INTEGER = Keywords.INTEGER.value
    STD_LOGIC = Keywords.STD_LOGIC.value
    SIGNED = Keywords.SIGNED.value


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
        return _wrap_in_IS_END_block(self.type, self.identifier, self._header())


class Entity(_DesignUnitForEntityAndComponent):
    def __init__(self, identifier: str):
        super().__init__(identifier, Keywords.ENTITY)


class ComponentDeclaration(_DesignUnitForEntityAndComponent):
    def __init__(self, identifier: str):
        super().__init__(identifier, Keywords.COMPONENT)


class Architecture:
    def __init__(self, identifier: str, design_unit: str):
        self.identifier = identifier
        self.design_unit = design_unit
        self._architecture_declaration_list = InterfaceList()
        self._architecture_component_list = InterfaceList()
        self._architecture_statement_part = None

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

    def __call__(self) -> Code:
        yield f"{Keywords.ARCHITECTURE.value} {self.identifier} {Keywords.OF.value} {self.design_unit} {Keywords.IS.value}"
        if len(self._architecture_declaration_list) > 0:
            yield from _indent_and_filter_non_empty_lines(
                _add_semicolons(
                    self._architecture_declaration_list(), semicolon_last=True
                )
            )
        if len(self._architecture_component_list) > 0:
            yield from _indent_and_filter_non_empty_lines(
                self._architecture_component_list()
            )
        yield f"{Keywords.BEGIN.value}"
        if self._architecture_statement_part:
            yield from _indent_and_filter_non_empty_lines(
                self._architecture_statement_part()
            )
        yield f"{Keywords.END.value} {Keywords.ARCHITECTURE.value} {self.identifier};"


class Process:
    def __init__(
            self,
            identifier: str,
            input_name: str,
            lookup_table_generator_function: CodeGenerator,
    ):
        self.identifier = identifier
        self._process_declaration_list = []
        self._process_statements_list = []
        self.lookup_table_generator_function = lookup_table_generator_function
        self.input = input_name

    @property
    def process_declaration_list(self):
        return self._process_declaration_list

    @process_declaration_list.setter
    def process_declaration_list(self, value: list[str]):
        self._process_declaration_list = value

    @property
    def process_statements_list(self):
        return self._process_statements_list

    @process_statements_list.setter
    def process_statements_list(self, value: list[str]):
        self._process_statements_list = value

    def _header(self) -> Code:
        if len(self.process_declaration_list) > 0:
            yield from _append_semicolons_to_lines(self._process_declaration_list)

    def _footer(self) -> Code:
        if len(self.process_statements_list) > 0:
            yield from _append_semicolons_to_lines(self._process_statements_list)
        yield from self.lookup_table_generator_function

    def __call__(self) -> Code:
        yield f"{self.identifier}_{Keywords.PROCESS.value}: {Keywords.PROCESS.value}({self.input})"
        yield from _indent_and_filter_non_empty_lines(self._header())
        yield f"{Keywords.BEGIN.value}"
        yield from _indent_and_filter_non_empty_lines(self._footer())
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
            variable_type: DataType,
            range: Optional[Union[str, int]],
            mode: Optional[Mode],
            value: Optional[Union[str, int]],
            declaration_type: Optional[str]

    ):
        self._identifier = identifier
        self._range = range
        self._variable_type = variable_type
        self._mode = mode
        self._value = value
        self._declaration_type = declaration_type

    @property
    def range(self) -> int:
        return self._range

    @range.setter
    def range(self, v: Optional[Union[str, int]]):
        self._range = v if v is not None else None

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, v):
        self._mode = v if v is not None else None

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, v):
        self._value = v if v is not None else None

    @property
    def declaration_type(self):
        return self._declaration_type

    @declaration_type.setter
    def declaration_type(self, v):
        self._declaration_type = v if v is not None else None

    def __call__(self) -> Code:
        range_part = "" if self._range is None else f"({self._range})"
        declaration_part = "" if self._declaration_type is None else f"{self._declaration_type} "
        value_part = "" if self.value is None else f" := {self.value}"
        mode_part = " " if self.mode is None else f" {self.mode.value} "
        yield from (
            f"{declaration_part}{self._identifier} :{mode_part}{self._variable_type.value}{range_part}{value_part}",
        )


class InterfaceSignal(InterfaceConstrained):
    def __init__(self, identifier: str, variable_type: DataType, range: Optional[Union[str, int]] = None,
                 mode: Optional[Mode] = None, value: Optional[Union[str, int]] = None):
        super().__init__(identifier, variable_type, range, mode, value, declaration_type="signal")


class InterfaceVariable(InterfaceConstrained):
    def __init__(self, identifier: str, variable_type: DataType, range: Optional[Union[str, int]] = None,
                 mode: Optional[Mode] = None, value: Optional[Union[str, int]] = None):
        super().__init__(identifier, variable_type, range, mode, value, declaration_type=None)


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


def _append_semicolons_to_lines(lines: Code) -> Code:
    temp = tuple(lines)
    yield from (f"{line};" for line in temp)


def _clause(clause_type: ClauseType, interfaces: Code) -> Code:
    yield f"{clause_type.value} ("
    yield from _indent_and_filter_non_empty_lines(_add_semicolons(interfaces))
    yield ");"


def _filter_empty_lines(lines: Code) -> Code:
    return filterfalse(_line_is_empty, lines)


def _line_is_empty(line: str) -> bool:
    return len(line) == 0


def _join_lines(lines) -> str:
    return "\n".join(lines)


def _empty_code_generator() -> Code:
    return []


def _indent_and_filter_non_empty_lines(lines: Code) -> Code:
    yield from map(indent, _filter_empty_lines(lines))


# noinspection PyPep8Naming
def _wrap_in_IS_END_block(
        block_type: Keywords, block_identifier: Identifier, lines: Code
) -> Code:
    yield f"{block_type.value} {block_identifier} {Keywords.IS.value}"
    yield from _indent_and_filter_non_empty_lines(lines)
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


def signal_assignment(identifier: str, statement):
    return f"{identifier} <= {statement};"
