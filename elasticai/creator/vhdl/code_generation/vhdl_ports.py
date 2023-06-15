from collections.abc import Callable, Sequence
from itertools import chain

from elasticai.creator.vhdl.design_base.ports import Port
from elasticai.creator.vhdl.design_base.signal import Signal


def signal_string(name: str, direction: str, width: int | str) -> str:
    if isinstance(width, int) and width > 0:
        data_type = f"std_logic"
    else:
        data_type = f"std_logic_vector({width} - 1 downto 0)"
    return f"{name} : {direction} {data_type}"


def _port_definition_from_port(
    p: Port, signal_width_handler: Callable[[Signal], str]
) -> list[str]:
    signals = []

    signals_and_directions = chain(zip(p.incoming, "in"), zip(p.outgoing, "out"))
    for signal, direction in signals_and_directions:
        width = signal_width_handler(signal)
        signals.append(signal_string(signal.name, direction, width))
    return wrap_lines_into_port_statement(signals)


def _extract_actual_width(s: Signal) -> str:
    if s.width == 0:
        return "std_logic"
    else:
        return f"std_logic_vector({s.width} - 1 downto 0)"


def _generate_template_width(s: Signal) -> str:
    if s.width == 0:
        return "std_logic"
    else:
        return f"std_logic_vector(${s.name}_width - downto 0)"


def vhdl_port_definition(p: Port) -> list[str]:
    return _port_definition_from_port(p, _extract_actual_width)


def template_string_for_port_definition(p: Port) -> list[str]:
    return _port_definition_from_port(p, _generate_template_width)


def wrap_lines_into_port_statement(lines: Sequence[str]) -> list[str]:
    semicolon_ended_lines = (f"{line};" for line in lines[:-1])
    last_line = lines[-1]
    return list(chain(["port ("], semicolon_ended_lines, [last_line, ");"]))


def expand_to_parameters_for_port_template(p: Port) -> dict[str, str]:
    return {
        f"{s.name}_width": f"{s.width}"
        for s in filter(lambda s: s.width > 0, p.signals)
    }
