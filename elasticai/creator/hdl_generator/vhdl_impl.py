"""VHDL implementation of HDL abstractions."""

from string import Template

from elasticai.creator.hdl_ir import Node
from elasticai.creator.ir2vhdl import (
    Instance,
    LogicSignal,
    LogicVectorSignal,
    NullDefinedLogicSignal,
    Signal,
)
from elasticai.creator.ir2vhdl.vhdl_template import EntityTemplateDirector

from .protocols import HDLInstance

# VHDL types already conform to HDL protocols via structural subtyping
# Export them directly without adapters


class VHDLTemplateDirector:
    """Wrapper for EntityTemplateDirector to match HDL protocol method names."""

    def __init__(self) -> None:
        self._director = EntityTemplateDirector()

    def set_prototype(self, prototype: str) -> "VHDLTemplateDirector":
        self._director.set_prototype(prototype)
        return self

    def add_parameter(self, name: str) -> "VHDLTemplateDirector":
        self._director.add_generic(name)
        return self

    def build(self) -> Template:
        return self._director.build()


def create_signal(name: str, width: int | None = None) -> Signal:
    """Create a VHDL signal.

    Args:
        name: The signal name.
        width: The signal width. If None or 1, creates a std_logic signal.
               Otherwise creates a std_logic_vector signal.

    Returns:
        A VHDL signal (Signal type).
    """
    if width is None or width == 1:
        return LogicSignal(name)
    else:
        return LogicVectorSignal(name, width)


def create_null_signal(name: str) -> Signal:
    """Create a signal that doesn't need to be defined (e.g., input ports).

    Args:
        name: The signal name.

    Returns:
        A null-defined VHDL signal (Signal type).
    """
    return NullDefinedLogicSignal(name)


def create_instance(
    node: Node,
    generics: dict[str, str],
    ports: dict[str, Signal],
) -> HDLInstance:
    """Create a VHDL entity instance.

    Args:
        node: The node representing the entity to instantiate.
        generics: Generic values (name -> value).
        ports: Port connections (port_name -> signal).

    Returns:
        A VHDL instance.
    """
    return Instance(node, generics, ports)
