"""Verilog implementation of HDL abstractions."""

from string import Template

from elasticai.creator.hdl_ir import Node
from elasticai.creator.ir2verilog.language import (
    BaseWire,
    Instance,
    NullWire,
    VectorWire,
    Wire,
)
from elasticai.creator.ir2verilog.templates import TemplateDirector

from .protocols import HDLInstance


# Verilog wire types already conform to HDLSignal protocol via structural subtyping
# Verilog Instance already conforms to HDLInstance protocol via structural subtyping
class VerilogTemplateDirector:
    """Wrapper for TemplateDirector to match HDL protocol method names."""

    def __init__(self) -> None:
        self._director = TemplateDirector()

    def set_prototype(self, prototype: str) -> "VerilogTemplateDirector":
        self._director.set_prototype(prototype)
        return self

    def add_parameter(self, name: str) -> "VerilogTemplateDirector":
        self._director.parameter(name)
        return self

    def build(self) -> Template:
        verilog_template = self._director.build()
        # VerilogTemplate has a substitute method, so we wrap it
        return _VerilogTemplateWrapper(verilog_template)


class _VerilogTemplateWrapper:
    """Wrapper to make VerilogTemplate compatible with string.Template interface."""

    def __init__(self, verilog_template: object) -> None:
        self._template = verilog_template

    def substitute(self, **kwargs: str | bool) -> str:
        """Substitute template parameters."""
        return self._template.substitute(kwargs)  # type: ignore


def create_signal(name: str, width: int | None = None) -> BaseWire:
    """Create a Verilog wire.

    Args:
        name: The wire name.
        width: The wire width. If None or 1, creates a single-bit wire.
               Otherwise creates a vector wire.

    Returns:
        A Verilog wire (BaseWire type).
    """
    if width is None or width == 1:
        return Wire(name)
    else:
        return VectorWire(name, width)


def create_null_signal(name: str) -> BaseWire:
    """Create a wire that doesn't need to be defined (e.g., input ports).

    Args:
        name: The wire name.

    Returns:
        A null wire (BaseWire type).
    """
    return NullWire(name)


def create_instance(
    node: Node,
    parameters: dict[str, str],
    ports: dict[str, BaseWire],
) -> HDLInstance:
    """Create a Verilog module instance.

    Args:
        node: The node representing the module to instantiate.
        parameters: Parameter values (name -> value).
        ports: Port connections (port_name -> wire).

    Returns:
        A Verilog instance that conforms to HDLInstance protocol.
    """
    return Instance(node, parameters, ports)
