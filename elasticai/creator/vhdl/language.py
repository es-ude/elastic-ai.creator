"""
The module contains classes and functions for generating vhdl code.
We provide code generators for the subset of vhdl that we need to implement
our neural network_component.py accelerators and test benches. We stick closely to the vhdl
formal grammar with our class names.

The core of this module is the `CodeGenerator`. Code generators are callables that return `Code`.
`Code` is an Iterable of strings. Depending on complexity, we define syntactic code_files of the vhdl
grammar as `CodeGenerator`s. The class can then be used to set up and configure a function that yields lines
of code as strings.
"""
from collections.abc import Iterable
from enum import Enum
from itertools import filterfalse
from typing import Callable, Iterator, Literal, Optional, Sequence, overload

from elasticai.creator.vhdl.code import Code, CodeGenerator, CodeGeneratorCompatible

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

    def code(self) -> Code:
        return self


class Entity(_DesignUnitForEntityAndComponent):
    def __init__(self, identifier: str):
        super().__init__(identifier, Keywords.ENTITY)


class ComponentDeclaration(_DesignUnitForEntityAndComponent):
    def __init__(self, identifier: str):
        super().__init__(identifier, Keywords.COMPONENT)


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


class InterfaceConstrained(Code, CodeGenerator):
    def __init__(
        self,
        identifier: str,
        identifier_type: DataType,
        width: Optional[str | int],
        mode: Optional[Mode],
        value: Optional[str | int],
        declaration_type: Optional[str],
    ):
        self._identifier = identifier
        self._range = f"({width})" if width else ""
        self._identifier_type = identifier_type
        self._mode = f" {mode.value} " if mode else " "
        self._value = f" := {value}" if value else ""
        self._declaration_type = f"{declaration_type} " if declaration_type else ""

    def __iter__(self) -> Iterator[str]:
        yield from (
            (
                f"{self._declaration_type}{self._identifier} :"
                f"{self._mode}{self._identifier_type.value}{self._range}{self._value}"
            ),
        )

    def code(self) -> Code:
        return self


class InterfaceVariable(InterfaceConstrained):
    def __init__(
        self,
        identifier: str,
        identifier_type: DataType,
        width: Optional[str | int] = None,
        mode: Optional[Mode] = None,
        value: Optional[str | int] = None,
    ):
        super().__init__(
            identifier, identifier_type, width, mode, value, declaration_type=None
        )


def _wrap_iterator_function_into_iterable(create_iterator: Callable[[], Iterator[str]]):
    class Wrapper(Iterable[str]):
        def __iter__(self) -> Iterator[str]:
            yield from create_iterator()

    wrapped = Wrapper()
    return wrapped


class CodeGeneratorConcatenation(Sequence[CodeGenerator], CodeGenerator):
    @overload
    def __getitem__(self, index: int) -> CodeGenerator:
        ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[CodeGenerator]:
        ...

    def __getitem__(self, index):
        if isinstance(index, slice):
            return CodeGeneratorConcatenation(self.interface_generators[index])
        else:
            return self.interface_generators[index]

    def __len__(self) -> int:
        return len(self.interface_generators)

    def __init__(self, *interfaces: CodeGeneratorCompatible):
        self.interface_generators: list[CodeGenerator] = [
            _unify_code_generators(interface) for interface in interfaces
        ]

    def append(self, interface: CodeGeneratorCompatible) -> None:
        self.interface_generators.append(_unify_code_generators(interface))

    def _code_iterator(self) -> Iterator[str]:
        for generator in self.interface_generators:
            for line in generator.code():
                yield line

    def code(self) -> Code:
        return _wrap_iterator_function_into_iterable(self._code_iterator)


class InterfaceSignal(InterfaceConstrained):
    def __init__(
        self,
        identifier: str,
        identifier_type: DataType,
        width: Optional[str | int] = None,
        mode: Optional[Mode] = None,
        value: Optional[str | int] = None,
    ) -> None:
        super().__init__(
            identifier, identifier_type, width, mode, value, declaration_type="signal"
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

    def code(self) -> Code:
        def generator():
            yield f"{self.map_name}: {self.component_name}"
            if len(self._generic_map_list) > 0:
                yield f"generic map ("
                yield from _filter_empty_lines(
                    _add_comma(self._generic_map_list.code())
                )
                yield ")"
            yield f"port map ("
            yield from _filter_empty_lines(_add_comma(self._signal_list.code()))
            yield ");"

        return generator()


class InterfaceList(CodeGeneratorConcatenation):
    pass


ClauseType = Literal[Keywords.GENERIC, Keywords.COMPONENT, Keywords.PORT]


def indent(line: str) -> str:
    return "".join(["\t", line])


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

    def _generator(self):
        yield f"procedure {self.identifier} ("
        if len(self._declaration_list) > 0:
            yield from _filter_empty_lines(
                _add_semicolons(self._declaration_list.code(), semicolon_last=True)
            )
        if len(self._declaration_list_with_is) > 0:
            yield from _filter_empty_lines(
                _add_is(self._declaration_list_with_is.code())
            )
        yield f"{Keywords.BEGIN.value}"
        if len(self._statement_list) > 0:
            yield from _filter_empty_lines(
                _add_semicolons(self._statement_list.code(), semicolon_last=True)
            )
        yield f"{Keywords.END.value} {self.identifier};"

    def code(self) -> Code:
        return self._generator()


def hex_representation(hex_value: str) -> str:
    return f'x"{hex_value}"'


def bin_representation(bin_value: str) -> str:
    return f'"{bin_value}"'
