from typing import Optional, cast

from elasticai.creator.vhdl.code_generation.addressable import calculate_address_width
from elasticai.creator.vhdl.code_generation.vhdl_ports import (
    template_string_for_port_definition,
)
from elasticai.creator.vhdl.design_base import std_signals as _signals
from elasticai.creator.vhdl.design_base.ports import Port


def create_port(
    x_width: int,
    y_width: int,
    *,
    x_count: int = 0,
    y_count: int = 0,
    x_address_width: Optional[int] = None,
    y_address_width: Optional[int] = None,
) -> Port:
    if (
        x_count == 0
        and y_count == 0
        and x_address_width is None
        and y_address_width is None
    ):
        return create_port_for_bufferless_design(x_width, y_width)
    else:
        return create_port_for_buffered_design(
            x_width=x_width,
            y_width=y_width,
            x_count=x_count,
            y_count=y_count,
            x_address_width=x_address_width,
            y_address_width=y_address_width,
        )


def create_port_for_bufferless_design(x_width: int, y_width: int) -> Port:
    in_signals = [
        _signals.enable(),
        _signals.clock(),
        _signals.x(x_width),
    ]
    out_signals = [
        _signals.y(y_width),
    ]
    return Port(incoming=in_signals, outgoing=out_signals)


def create_port_for_buffered_design(
    *,
    x_width: int,
    y_width: int,
    x_count: int,
    y_count: int,
    x_address_width: Optional[int] = None,
    y_address_width: Optional[int] = None,
) -> Port:
    if not (
        (not (x_address_width is None and y_address_width is None))
        or (x_count > 0 or y_count > 0)
    ):
        raise ValueError(
            "Provide either (x_count, y_count) or (x_address_width, y_address_width)"
        )
    if x_address_width is None and y_address_width is None:
        x_address_width = calculate_address_width(x_count)
        y_address_width = calculate_address_width(y_count)

    in_signals = [
        _signals.enable(),
        _signals.clock(),
        _signals.x(x_width),
        _signals.y_address(cast(int, y_address_width)),
    ]
    out_signals = [
        _signals.done(),
        _signals.y(y_width),
        _signals.x_address(cast(int, x_address_width)),
    ]
    return Port(incoming=in_signals, outgoing=out_signals)


def port_definition_template_for_buffered_design() -> list[str]:
    return template_string_for_port_definition(
        create_port_for_buffered_design(x_width=1, y_width=1, x_count=1, y_count=1)
    )


def port_definition_template_for_bufferless_design() -> list[str]:
    return template_string_for_port_definition(create_port_for_bufferless_design(1, 1))
