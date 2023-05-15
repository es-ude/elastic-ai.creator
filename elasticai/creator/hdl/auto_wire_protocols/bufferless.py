from elasticai.creator.hdl.code_generation.vhdl_ports import (
    template_string_for_port_definition,
)
from elasticai.creator.hdl.design_base import std_signals as _signals
from elasticai.creator.hdl.design_base.ports import Port


def create_port_for_bufferless_design(x_width: int, y_width: int):
    return Port(
        incoming=[
            _signals.enable(),
            _signals.clock(),
            _signals.x(x_width),
        ],
        outgoing=[_signals.y(y_width)],
    )


def create_port_template_for_bufferless_design() -> list[str]:
    return template_string_for_port_definition(create_port_for_bufferless_design(1, 1))
