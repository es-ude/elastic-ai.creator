from dataclasses import dataclass

from elasticai.creator.vhdl.signals import Signal
from elasticai.creator.vhdl.signals.default_signal_definition import (
    generate_default_suffix,
)
from elasticai.creator.vhdl.signals.signal_configuration import _SignalConfiguration


@dataclass(kw_only=True)
class _VectorSignalConfiguration(_SignalConfiguration):
    width: int


class _VectorSignalImpl:
    def __init__(self, config: _VectorSignalConfiguration) -> None:
        self._accepted_names = config.accepted_names
        self._id = config.id
        self._width = config.width
        self._default = generate_default_suffix(config.default)

    def id(self) -> str:
        return self._id

    def definition(self, prefix: str = "") -> str:
        return (
            f"signal {prefix}{self.id()} :"
            f" std_logic_vector({self._width - 1} downto 0){self._default};"
        )

    def accepts(self, other: Signal) -> bool:
        return (
            isinstance(other, self.__class__)
            and other.id() in self._accepted_names
            and self._width == other._width
        )
