from elasticai.creator.vhdl.language.signals._signal_impl import _SignalImpl
from elasticai.creator.vhdl.language.signals._vector_signal_impl import (
    _VectorSignalImpl,
)


class Signal:
    def __init__(self, *, id: str, accepted_names: list[str], width: int):
        if width == 0:
            self._wrapped: _SignalImpl | _VectorSignalImpl = _SignalImpl(
                id=id, accepted_names=accepted_names, default="'0'"
            )
        else:
            self._wrapped = _VectorSignalImpl(
                id=id,
                accepted_names=accepted_names,
                default="(other => '0')",
                width=width,
            )

    def accepts(self, other: "Signal") -> bool:
        return self._wrapped.accepts(other._wrapped)

    def id(self) -> str:
        return self._wrapped.id()

    def definition(self, prefix: str = "") -> str:
        return self._wrapped.definition(prefix)
