"""
The module contains classes and functions for generating vhdl code.
We provide code generators for the subset of vhdl that we need for implementing
our neural network accelerators and test benches. We stick closely to the vhdl
formal grammar with our class names.
"""
from enum import Enum
from itertools import filterfalse
from typing import Callable, Iterable, Union, NewType, Literal

Identifier = str
Code = Iterable[str]
CodeGenerator = Callable[[], Code]


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


Mode = Literal[
    Keywords.IN, Keywords.OUT, Keywords.INOUT, Keywords.BUFFER, Keywords.BUFFER
]


class _DesignEntity:
    def __init__(
        self, identifier: str, design_type: Literal[Keywords.ENTITY, Keywords.PORT]
    ):
        self.identifier = identifier
        self.generic_list = []
        self.port_list = []
        self.type = design_type

    def _header(self) -> Code:
        if len(self.generic_list) > 0:
            yield from _clause(Keywords.GENERIC, self.generic_list)
        if len(self.port_list) > 0:
            yield from _clause(Keywords.PORT, self.port_list)

    def __call__(self) -> Code:
        return wrap_in_IS_END_block(self.type, self.identifier, self._header())


class EntityDeclaration(_DesignEntity):
    def __init__(self, identifier: str):
        super().__init__(identifier, Keywords.ENTITY)


class ComponentDeclaration(_DesignEntity):
    def __init__(self, identifier: str):
        super().__init__(identifier, Keywords.COMPONENT)


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


def _add_semicolons(lines: Code) -> Code:
    temp = tuple(lines)
    yield from (f"{line};" for line in temp[:-1])
    yield f"{temp[-1]}"


def _clause(clause_type: ClauseType, interfaces: Code) -> Code:
    yield f"{clause_type.value} ("
    yield from indent_and_filter_non_empty_lines(_add_semicolons(interfaces))
    yield ");"


class InterfaceVariableDeclaration:
    def __init__(self, identifier: str, variable_type: str, value: str):
        self.identifier = identifier
        self.value = value
        self.variable_type = variable_type

    def __call__(self) -> Code:
        return (f"{self.identifier} : {self.variable_type} := {self.value}",)


def indent(line: str) -> str:
    return "".join(["\t", line])


def _filter_empty_lines(lines: list[str]) -> list[str]:
    return [line for line in lines if line != ""]


def _line_is_empty(line: str) -> bool:
    return len(line) == 0


def _join_lines(lines):
    return "\n".join(lines)


def _empty_code_generator() -> list[str]:
    return []


def indent_and_filter_non_empty_lines(lines: Code) -> Code:
    yield from map(indent, filterfalse(_line_is_empty, lines))


# noinspection PyPep8Naming
def wrap_in_IS_END_block(
    block_type: Keywords, block_identifier: Identifier, lines: Code
) -> Code:
    yield f"{block_type.value} {block_identifier} {Keywords.IS.value}"
    yield from indent_and_filter_non_empty_lines(lines)
    yield f"{Keywords.END.value} {block_type.value} {block_identifier};"
