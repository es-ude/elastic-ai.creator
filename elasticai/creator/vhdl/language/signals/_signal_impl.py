from ._signal import Signal
from ._signal_base import _SignalBase
from ._signal_base import _SignalBaseConfiguration as _SignalConfiguration


class _SignalImpl(_SignalBase):
    def definition(self, prefix: str = "") -> str:
        return f"signal {prefix}{self.id()} : std_logic{self._default};"

    def accepts(self, other: "Signal") -> bool:
        return isinstance(other, self.__class__) and self.id() in self._accepted_names