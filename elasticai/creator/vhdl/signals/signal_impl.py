from elasticai.creator.vhdl.signals import Signal
from elasticai.creator.vhdl.signals.default_signal_definition import (
    generate_default_suffix,
)
from elasticai.creator.vhdl.signals.signal_configuration import _SignalConfiguration


class _SignalImpl(Signal):
    def __init__(self, config: _SignalConfiguration):
        self._id = config.id
        self._default = generate_default_suffix(config.default)
        self._accepted_names = config.accepted_names

    def id(self) -> str:
        return self._id

    def definition(self, prefix: str = "") -> str:
        return f"signal {prefix}{self.id()} : std_logic{self._default};"

    def accepts(self, other: Signal) -> bool:
        return isinstance(other, self.__class__) and other.id() in self._accepted_names
