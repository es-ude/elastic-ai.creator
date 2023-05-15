from elasticai.creator.hdl.code_generation.code_generation import (
    calculate_address_width,
)
from elasticai.creator.hdl.code_generation.vhdl_ports import (
    template_string_for_port_definition,
)
from elasticai.creator.hdl.design_base import std_signals as _signals
from elasticai.creator.hdl.design_base.ports import Port


def create_port_for_buffered_design(
    x_width: int, y_width: int, x_count: int, y_count: int
) -> Port:
    in_signals = [
        _signals.enable(),
        _signals.clock(),
        _signals.x(x_width),
        _signals.y_address(calculate_address_width(y_count)),
    ]
    out_signals = [
        _signals.done(),
        _signals.y(y_width),
        _signals.x_address(calculate_address_width(x_count)),
    ]
    return Port(incoming=in_signals, outgoing=out_signals)


def port_definition_template_for_buffered_design() -> list[str]:
    return template_string_for_port_definition(
        create_port_for_buffered_design(1, 1, 1, 1)
    )
