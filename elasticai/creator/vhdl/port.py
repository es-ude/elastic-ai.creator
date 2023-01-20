from abc import abstractmethod
from collections.abc import Sequence
from itertools import chain
from typing import Any, Protocol, TypeVar, Union, runtime_checkable

from elasticai.creator.vhdl.code import Code, CodeGenerator
from elasticai.creator.vhdl.connectable import Connectable
from elasticai.creator.vhdl.signals import (
    LogicInSignal,
    LogicInVectorSignal,
    LogicOutSignal,
    LogicOutVectorSignal,
)
from elasticai.creator.vhdl.typing import Identifiable

T_Signal = TypeVar(
    "T_Signal",
    bound=Union[
        LogicInSignal, LogicInVectorSignal, LogicOutVectorSignal, LogicOutSignal
    ],
)


@runtime_checkable
class SignalProvider(Protocol):
    @abstractmethod
    def in_signals(self) -> Sequence[LogicInSignal | LogicInVectorSignal]:
        ...

    @abstractmethod
    def out_signals(self) -> Sequence[LogicOutSignal | LogicOutVectorSignal]:
        ...


class _BaseConnectableSignalProvider(Connectable, SignalProvider):
    def connect(self, other: Any):
        self._connect_signal_providers(other)
        if isinstance(other, _BaseConnectableSignalProvider):
            other._connect_signal_providers(self)

    def is_missing_inputs(self) -> bool:
        return any((signal.is_missing_inputs() for signal in self.in_signals()))

    @staticmethod
    def _connect_signals(
        ins: Sequence[LogicInSignal | LogicInVectorSignal],
        outs: Sequence[LogicOutSignal | LogicOutVectorSignal],
    ):
        for signal in ins:
            for out_signal in outs:
                signal.connect(out_signal)
            if not signal.is_missing_inputs():
                break

    def __init__(
        self,
        in_signals: Sequence[LogicInSignal | LogicInVectorSignal],
        out_signals: Sequence[LogicOutSignal | LogicOutVectorSignal],
    ):
        self._in_signals = in_signals
        self._out_signals = out_signals

    def in_signals(self) -> Sequence[LogicInSignal | LogicInVectorSignal]:
        return self._in_signals

    def out_signals(self) -> Sequence[LogicOutSignal | LogicOutVectorSignal]:
        return self._out_signals

    def _connect_signal_providers(self, right: Any):
        if isinstance(right, SignalProvider):
            self._connect_signals(self.in_signals(), right.out_signals())


class PortMap(_BaseConnectableSignalProvider, CodeGenerator, Identifiable):
    def __init__(
        self,
        id: str,
        in_signals: Sequence[LogicInSignal | LogicInVectorSignal],
        out_signals: Sequence[LogicOutSignal | LogicOutVectorSignal],
    ):
        self._in_signal_ids_without_prefixes = list(
            signal.id() for signal in in_signals
        )
        self._out_signal_ids_without_prefixes = list(
            signal.id() for signal in out_signals
        )
        super().__init__(
            in_signals=self._prefix_signals(id, in_signals),
            out_signals=self._prefix_signals(id, out_signals),
        )
        self._id = id

    @staticmethod
    def _prefix_signals(id, signals):
        return tuple(signal.with_prefix(id) for signal in signals)

    def id(self) -> str:
        return self._id

    def signal_definitions(self) -> Code:
        return [
            signal.definition()
            for signal in chain(self.in_signals(), self.out_signals())
        ]

    def code(self) -> Code:
        result = [
            f"{self.id()} : entity work.{self.id()}(rtl)",
            "port map (",
        ]
        for map_signal, port_signal in chain(
            zip(self.in_signals(), self._in_signal_ids_without_prefixes),
            zip(self.out_signals(), self._out_signal_ids_without_prefixes),
        ):
            result.append(f"{port_signal} => {map_signal.id()},")
        result[-1] = result[-1].strip(",")
        result.append(");")
        return result

    def code_for_signal_connections(self) -> Code:
        return tuple(line for signal in self.in_signals() for line in signal.code())


class Port(_BaseConnectableSignalProvider):
    def build_port_map(self, id: str) -> PortMap:
        pm = PortMap(
            id=id,
            in_signals=[signal.reverse() for signal in self.out_signals()],
            out_signals=[signal.reverse() for signal in self.in_signals()],
        )
        return pm
