from abc import abstractmethod
from typing import Optional, Protocol

from elasticai.creator.vhdl.signals.base_signal import Identifiable


def _generate_default_assignment(default: Optional[str]) -> str:
    if default is None:
        return ""
    return f" := {default}"


def create_std_logic_definition(
    signal: Identifiable, default: Optional[str] = None
) -> str:
    default = _generate_default_assignment(default)
    return f"signal {signal.id()} : std_logic{default};"


class HasWidth(Protocol):
    @abstractmethod
    def width(self) -> int:
        ...


class Vector(Identifiable, HasWidth, Protocol):
    ...


def create_std_logic_vector_definition(
    signal: Vector, default: Optional[str] = None
) -> str:
    default = _generate_default_assignment(default)
    return (
        f"signal {signal.id()} : std_logic_vector({signal.width() - 1} downto 0)"
        f"{default};"
    )
