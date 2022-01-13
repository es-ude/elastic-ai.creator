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


class DataType(Enum):
    INTEGER = Keywords.INTEGER.value
    STD_LOGIC = Keywords.STD_LOGIC.value


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


class InterfaceVariable:
    def __init__(
        self,
        identifier: str,
        variable_type: DataType,
        value: Optional[Union[str, int]] = None,
        mode: Optional[Mode] = None,
    ):
        self.identifier = identifier
        self.value = value
        self.variable_type = variable_type
        self.mode = mode

    @property
    def value(self) -> int:
        return self._value

    @value.setter
    def value(self, v: Optional[Union[str, int]]):
        self._value = int(v) if v is not None else None

    def __call__(self) -> Code:
        value_part = "" if self.value is None else f" := {self.value}"
        mode_part = "" if self.mode is None else f" {self.mode} "
        yield from (
            f"{self.identifier} : {mode_part}{self.variable_type.value}{value_part}",
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


def _add_semicolons(lines: Code) -> Code:
    temp = tuple(lines)
    yield from (f"{line};" for line in temp[:-1])
    yield f"{temp[-1]}"


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
