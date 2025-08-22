from collections.abc import Sequence

from elasticai.creator.vhdl.design.signal import Signal


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


def create_connections_using_to_from_pairs(mapping: dict[str, str]) -> list[str]:
    connections: list[str] = []
    for _to, _from in mapping.items():
        connections.append(create_connection(_to, _from))
    return connections


def create_connection(dst, source) -> str:
    return f"{dst} <= {source};"


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


def to_vhdl_binary_string(number: int, number_of_bits: int) -> str:
    max_val = 2 ** (number_of_bits - 1)
    if (number < -max_val) | (number > max_val - 1):
        raise ValueError(
            f"Value '{number}' cannot be represented with {number_of_bits} bits."
        )
    if number < 0:
        twos = (1 << number_of_bits) + number
    else:
        twos = number
    return f'"{twos:0{number_of_bits}b}"'
