import re
from abc import abstractmethod
from typing import Protocol, Sequence, overload

from elasticai.creator.hdl.code_generation.code_generation import to_hex


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


def create_connections(mapping: dict[str, str]) -> list[str]:
    mapping = _sorted_dict(mapping)
    connections: list[str] = []
    for _to, _from in mapping.items():
        connections.append(f"{_to} <= {_from};")
    return connections


class Signal(Protocol):
    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def width(self) -> int:
        ...


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
            f"signal {name} : std_logic_vector({width - 1} downto 0) := ('other' =>"
            " '0');"
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
def extract_rom_values(text: str) -> tuple[str]:
    ...


@overload
def extract_rom_values(text: list[str]) -> tuple[str]:
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
