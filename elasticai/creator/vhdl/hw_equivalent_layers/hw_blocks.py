import dataclasses
from abc import abstractmethod
from enum import auto, Enum
from typing import Protocol, Iterable, Any

from vhdl.code import Code, Translatable


class HWBlockInterface(Protocol):
    @property
    @abstractmethod
    def x_width(self) -> int:
        ...

    @property
    @abstractmethod
    def y_width(self) -> int:
        ...

    @abstractmethod
    def signal_definitions(self, prefix: str) -> Code:
        ...

    @abstractmethod
    def instantiation(self, prefix: str) -> Code:
        ...


class BufferedHWBlockInterface(HWBlockInterface, Protocol):
    @property
    @abstractmethod
    def x_address_width(self) -> int:
        ...

    @property
    @abstractmethod
    def y_address_width(self) -> int:
        ...


class TranslatableHWBlockInterface(HWBlockInterface, Translatable, Protocol):
    ...


class TranslatableBufferedHWBlock(BufferedHWBlockInterface, Translatable, Protocol):
    ...


class StrEnum(str, Enum):
    def _generate_next_value_(
        name: str, start: int, count: int, last_values: list[Any]
    ) -> Any:
        return name.lower()

    def __str__(self):
        return self.value


class StdLogicInSignals(StrEnum):
    CLOCK = auto()
    ENABLE = auto()


class LogicVectorInSignals(StrEnum):
    X = auto()
    Y_ADDRESS = auto()


class LogicVectorOutSignals(StrEnum):
    Y = auto()
    X_ADDRESS = auto()


class StdLogicOutSignals(StrEnum):
    DONE = auto()
    CLOCK = auto()


SIGNAL_MAPPING = (
    (StdLogicOutSignals.CLOCK, StdLogicInSignals.CLOCK),
    (StdLogicOutSignals.DONE, StdLogicInSignals.ENABLE),
    (LogicVectorOutSignals.Y, LogicVectorInSignals.X),
    (LogicVectorOutSignals.X_ADDRESS, LogicVectorInSignals.Y_ADDRESS),
)


class BaseHWBlockInterface(HWBlockInterface):
    _vector_signal_width_mapping = (
        LogicVectorInSignals.X,
        LogicVectorOutSignals.Y,
    )
    _logic_signals = (StdLogicInSignals.ENABLE, StdLogicInSignals.CLOCK)

    def __init__(self, x_width: int, y_width: int):
        self._x_data_width = x_width
        self._y_data_width = y_width

    @property
    def y_width(self) -> int:
        return self._y_data_width

    @property
    def x_width(self) -> int:
        return self._x_data_width

    def _get_vector_signal_name_to_signal_width_mapping(
        self,
    ) -> Iterable[tuple[str, int]]:
        return map(
            lambda s: (str(s), getattr(self, f"{s}_width")),
            self._vector_signal_width_mapping,
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
        yield from _ComponentInstantiation(prefix=prefix).code()


class BufferedBaseHWBlockInterface(BaseHWBlockInterface, BufferedHWBlockInterface):
    _vector_signal_width_mapping = (
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
        self._y_data_width = y_width
        self._x_data_width = x_width
        self._x_address_width = x_address_width
        self._y_address_width = y_address_width

    @property
    def x_address_width(self) -> int:
        return self._x_address_width

    @property
    def y_address_width(self) -> int:
        return self._y_address_width

    def instantiation(self, prefix: str) -> Code:
        yield from _BufferedComponentInstantiation(prefix=prefix).code()


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


@dataclasses.dataclass
class _ComponentInstantiation:
    prefix: str

    def _generate_port_map_signals(self) -> list[str]:
        return [
            self._generate_port_to_signal_connection(port)
            for port in (
                "enable",
                "clock",
                "x",
                "y",
            )
        ]

    def _generate_port_to_signal_connection(self, port: str) -> str:
        return f"{port} => {self.prefix}_{port},"

    @staticmethod
    def _remove_comma_from_last_signal(signals: list[str]) -> list[str]:
        signals[-1] = signals[-1][:-1]
        return signals

    def code(self) -> Code:
        name = self.prefix
        code = [
            f"{name} : entity work.{name}(rtl)",
            "port map(",
        ]
        mapping = self._generate_port_map_signals()
        mapping = self._remove_comma_from_last_signal(mapping)
        code.extend(mapping)
        code.append(");")
        return code


class _BufferedComponentInstantiation(_ComponentInstantiation):
    def _generate_port_map_signals(self) -> list[str]:
        signals = super()._generate_port_map_signals()
        signals.extend(
            self._generate_port_to_signal_connection(port)
            for port in ("x_address", "y_address", "done")
        )
        return signals
