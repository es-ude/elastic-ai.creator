import re
from dataclasses import dataclass
from typing import Iterable, Sequence, overload

from elasticai.creator.hdl.code_generation.code_generation import to_hex
from elasticai.creator.hdl.code_generation.parsing import parse
from elasticai.creator.hdl.code_generation.tokens import Token
from elasticai.creator.hdl.design_base.signal import Signal as _Signal
from elasticai.creator.hdl.vhdl.code_generation.tokens import tokenize, tokenize_rule


def _sorted_dict(items: dict[str, str]) -> dict[str, str]:
    return dict((key, items[key]) for key in sorted(items))


def create_instance(
    *,
    name: str,
    entity: str,
    signal_mapping: dict[str, str],
    library: str,
    architecture: str = "rtl",
) -> list[str]:
    signal_mapping = _sorted_dict(signal_mapping)
    result = [f"{name} : entity {library}.{entity}({architecture})", "port map("]
    for _from in tuple(signal_mapping.keys())[:-1]:
        _to = signal_mapping[_from]
        result.append(f"  {_from} => {_to},")
    last_from = tuple(signal_mapping.keys())[-1]
    last_to = signal_mapping[last_from]
    result.append(f"  {last_from} => {last_to}")
    result.append(");")
    return result


@dataclass(eq=True, frozen=True)
class EntityInstance:
    name: str
    entity: str
    signal_mapping: dict[str, str]
    library: str
    architecture: str = "rtl"

    def to_code(self) -> list[str]:
        return create_instance(
            name=self.name,
            entity=self.entity,
            signal_mapping=self.signal_mapping,
            library=self.library,
            architecture=self.architecture,
        )

    def from_code(self):
        rule = (
            "ID ':' entity ID '.' ID '(' ID ')' 'port' 'map' '(' port_assignment_list"
            " ')' ';'"
        )
        assignment_list_rules = ["port_assignment_list ';' ID '=>' ID", "ID '=>' ID"]


@dataclass(eq=True, frozen=True)
class AssignmentList:
    sources: tuple[str, ...]
    sinks: tuple[str, ...]

    def to_dict(self) -> dict[str, str]:
        return dict(zip(self.sinks, self.sources))

    def to_code(self) -> list[str]:
        connections: list[str] = []
        for _to, _from in zip(self.sinks, self.sources):
            connections.append(f"{_to} <= {_from};")
        return connections

    @classmethod
    def from_code(cls, code: Iterable[str]) -> "AssignmentList":
        assignment = tokenize_rule("assignment: ID '<=' ID ';'")
        result: dict[str, str] = {}

        tokens = tokenize(code)

        def handle_completion(tokens: Sequence[Token]):
            result[tokens[0].value] = tokens[2].value

        parse(tokens, (assignment,), handle_completion)
        return cls.from_dict(result)

    @classmethod
    def from_dict(cls, mapping: dict[str, str]) -> "AssignmentList":
        return cls(sinks=tuple(mapping.keys()), sources=tuple(mapping.values()))


def create_connections_using_to_from_pairs(mapping: dict[str, str]) -> list[str]:
    mapping = _sorted_dict(mapping)
    connections: list[str] = []
    for _to, _from in mapping.items():
        connections.append(f"{_to} <= {_from};")
    return connections


@dataclass(eq=True, frozen=True)
class Signal(_Signal):
    def define(self) -> str:
        return signal_definition(name=self.name, width=self.width)

    @classmethod
    def from_definition(cls, code: str):
        tokens = tuple(tokenize(code))
        if tokens[3].value == "std_logic_vector":
            width = int(tokens[5].value)
            if tokens[6].value == "-":
                width = width - int(tokens[7].value)
        else:
            width = 0
        name = tokens[1].value
        return cls(name=name, width=width)


@dataclass
class SignalDefinitionList:
    signals: set[Signal]

    def to_code(self) -> list[str]:
        return [signal_definition(name=s.name, width=s.width) for s in self.signals]

    @classmethod
    def from_code(cls, code: Iterable[str]):
        _rule_start = "signal_definition: 'signal' ID ':'"
        _vector_end = "'downto' '0' ')'  ':' '=' '(' 'others' '=>' ''' '0' ''' ')' ';'"
        rules = (
            tokenize_rule(r)
            for r in [
                f"{_rule_start} 'std_logic' ':' '=' ''0'' ';'",
                f"{_rule_start} 'std_logic_vector' '(' NUMBER '-' '1' {_vector_end}",
                f"{_rule_start} 'std_logic_vector' '(' NUMBER {_vector_end}",
            ]
        )
        signals = set()

        def handle_completion(parsed_tokens):
            if Token("OPERATOR", "-") in parsed_tokens:
                width = int(parsed_tokens[5].value)
            elif Token("ID", "std_logic_vector") in parsed_tokens:
                width = int(parsed_tokens[5].value) + 1
            else:
                width = 0
            name = parsed_tokens[1].value
            signals.add(Signal(name=name, width=width))

        parse(tokenize(code), tuple(rules), handle_completion)

        return SignalDefinitionList(signals)


def create_signal_definitions(prefix: str, signals: Sequence[Signal]):
    return sorted(
        [
            signal_definition(name=f"{prefix}{signal.name}", width=signal.width)
            for signal in signals
        ]
    )


def signal_definition(
    *,
    name: str,
    width: int,
):
    def vector_signal(name: str, width) -> str:
        return (
            f"signal {name} : std_logic_vector({width - 1} downto 0)"
            " := (others => '0');"
        )

    def logic_signal(name: str) -> str:
        return f"signal {name} : std_logic := '0';"

    if width > 0:
        return vector_signal(name, width)
    else:
        return logic_signal(name)


def hex_representation(hex_value: str) -> str:
    return f'x"{hex_value}"'


def bin_representation(bin_value: str) -> str:
    return f'"{bin_value}"'


def to_vhdl_hex_string(number: int, bit_width: int) -> str:
    return f"'x{to_hex(number, bit_width)}'"


def generate_hex_for_rom(value: str):
    return f'x"{value}"'


@overload
def extract_rom_values(text: str) -> tuple[str, ...]:
    ...


@overload
def extract_rom_values(text: list[str]) -> tuple[str, ...]:
    ...


def extract_rom_values(text: str | list[str]) -> tuple[str, ...]:
    if not isinstance(text, list):
        text = [text]
    values: tuple[str, ...] = tuple()
    for line in text:
        match = re.match(
            r'.*\(x"([a-f0-9]+(",\s?x"[a-f0-9]+)*)"\)',
            line,
        )
        if match is not None:
            array = match.group(1)
            values = tuple(re.split(r'(?:",\s?x")', array))

    return values
