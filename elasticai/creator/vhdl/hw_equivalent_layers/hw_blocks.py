from abc import abstractmethod
from enum import Enum, auto
from itertools import chain
from typing import Any, Iterable, Protocol

from elasticai.creator.vhdl.code import Code, Translatable


class HWBlockInterface(Protocol):
    @abstractmethod
    def signal_definitions(self, prefix: str) -> Code:
        ...

    @abstractmethod
    def instantiation(self, prefix: str) -> Code:
        ...


class TranslatableHWBlockInterface(HWBlockInterface, Translatable, Protocol):
    ...


class SignalEnum(str, Enum):
    def _generate_next_value_(
        name: str, start: int, count: int, last_values: list[Any]
    ) -> Any:
        return name.lower()

    def __str__(self):
        return self.value


class StdLogicInSignals(SignalEnum):
    CLOCK = auto()
    ENABLE = auto()


class LogicVectorInSignals(SignalEnum):
    X = auto()
    Y_ADDRESS = auto()


class LogicVectorOutSignals(SignalEnum):
    Y = auto()
    X_ADDRESS = auto()


class StdLogicOutSignals(SignalEnum):
    DONE = auto()
    CLOCK = auto()


SIGNAL_MAPPING = (
    (StdLogicOutSignals.CLOCK, StdLogicInSignals.CLOCK),
    (StdLogicOutSignals.DONE, StdLogicInSignals.ENABLE),
    (LogicVectorOutSignals.Y, LogicVectorInSignals.X),
    (LogicVectorOutSignals.X_ADDRESS, LogicVectorInSignals.Y_ADDRESS),
)


class BaseHWBlock(HWBlockInterface):
    _vector_signals: tuple[SignalEnum, ...] = (
        LogicVectorInSignals.X,
        LogicVectorOutSignals.Y,
    )
    _logic_signals: tuple[SignalEnum, ...] = (
        StdLogicInSignals.ENABLE,
        StdLogicInSignals.CLOCK,
    )

    def __init__(self, x_width: int, y_width: int):
        self._x_width = x_width
        self._y_width = y_width

    def _get_vector_signal_name_to_signal_width_mapping(
        self,
    ) -> Iterable[tuple[str, int]]:
        return map(
            lambda s: (str(s), getattr(self, f"_{s}_width")),
            self._vector_signals,
        )

    def _get_logic_signal_names(self) -> Iterable[str]:
        return map(str, self._logic_signals)

    def signal_definitions(self, prefix: str) -> Code:
        yield from generate_signal_definitions(
            prefix,
            std_logic_signals=self._get_logic_signal_names(),
            logic_vector_signals=self._get_vector_signal_name_to_signal_width_mapping(),
        )

    def instantiation(self, prefix: str) -> Code:
        yield from instantiate_component(
            prefix=prefix, signals=chain(self._vector_signals, self._logic_signals)
        )


class BufferedBaseHWBlock(BaseHWBlock):
    _vector_signals = (
        LogicVectorInSignals.X,
        LogicVectorOutSignals.Y,
        LogicVectorOutSignals.X_ADDRESS,
        LogicVectorInSignals.Y_ADDRESS,
    )
    _logic_signals = (
        StdLogicInSignals.ENABLE,
        StdLogicInSignals.CLOCK,
        StdLogicOutSignals.DONE,
    )

    def __init__(
        self,
        x_width: int,
        y_width: int,
        x_address_width: int,
        y_address_width: int,
    ):
        super().__init__(y_width=y_width, x_width=x_width)
        self._x_address_width = x_address_width
        self._y_address_width = y_address_width


def generate_signal_definitions(
    prefix: str,
    std_logic_signals: Iterable[str],
    logic_vector_signals: Iterable[tuple[str, int]],
) -> Code:
    def _logic_vector(suffix, width) -> str:
        return f"signal {prefix}_{suffix} : std_logic_vector({width - 1} downto 0);"

    def _std_logic(suffix) -> str:
        return f"signal {prefix}_{suffix} : std_logic := '0';"

    code = [_std_logic(suffix) for suffix in std_logic_signals]
    code.extend(
        [_logic_vector(suffix, width) for suffix, width in logic_vector_signals]
    )

    return code


def instantiate_component(prefix: str, signals: Iterable[SignalEnum]) -> Code:
    def _generate_port_to_signal_connection(port: SignalEnum) -> str:
        return f"{port} => {prefix}_{port},"

    def _remove_comma_from_last_signal(signals: list[str]) -> list[str]:
        signals[-1] = signals[-1][:-1]
        return signals

    def _generate_port_map_signals(signals) -> list[str]:
        return [_generate_port_to_signal_connection(port) for port in signals]

    name = prefix
    code = [
        f"{name} : entity work.{name}(rtl)",
        "port map(",
    ]
    mapping = _generate_port_map_signals(signals)
    mapping = _remove_comma_from_last_signal(mapping)
    code.extend(mapping)
    code.append(");")
    return code
