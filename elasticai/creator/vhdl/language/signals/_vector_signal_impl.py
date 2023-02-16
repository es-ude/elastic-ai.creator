from dataclasses import dataclass

from ._signal import Signal
from ._signal_base import _SignalBase, _SignalBaseConfiguration


@dataclass(kw_only=True)
class _VectorSignalConfiguration(_SignalBaseConfiguration):
    width: int


class _VectorSignalImpl(_SignalBase):
    def __init__(self, config: _VectorSignalConfiguration) -> None:
        super().__init__(config)
        self._width = config.width

    def definition(self, prefix: str = "") -> str:
        return (
            f"signal {prefix}{self.id()} :"
            f" std_logic_vector({self._width - 1} downto 0){self._default};"
        )

    def accepts(self, other: Signal) -> bool:
        if isinstance(other, _VectorSignalImpl):
            return other.id() in self._accepted_names and self._width == other._width
        return False
