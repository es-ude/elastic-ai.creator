import dataclasses
from abc import abstractmethod
from typing import Protocol

from vhdl.code import Code, Translatable


class HWBlockInterface(Protocol):
    @property
    @abstractmethod
    def data_width(self) -> int:
        ...

    @abstractmethod
    def signals(self, prefix: str) -> Code:
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


class BaseHWBlockInterfaceInterface(HWBlockInterface):
    def __init__(self, data_width: int):
        self._data_width = data_width

    @property
    def data_width(self) -> int:
        return self._data_width

    def signals(self, prefix: str) -> Code:
        yield from _SignalsForBufferlessComponent(
            prefix=prefix, data_width=self.data_width
        ).code()

    def instantiation(self, prefix: str) -> Code:
        yield from _ComponentInstantiation(prefix=prefix).code()


class BufferedBaseHWBlockInterface(BufferedHWBlockInterface):
    def __init__(self, data_width: int, x_address_width: int, y_address_width: int):
        self._data_width = data_width
        self._x_address_width = x_address_width
        self._y_address_width = y_address_width

    @property
    def data_width(self) -> int:
        return self._data_width

    @property
    def x_address_width(self) -> int:
        return self._x_address_width

    @property
    def y_address_width(self) -> int:
        return self._y_address_width

    def signals(self, prefix: str) -> Code:
        yield from _SignalsForComponentWithBuffer(
            data_width=self.data_width,
            prefix=prefix,
            x_address_width=self.x_address_width,
            y_address_width=self.y_address_width,
        ).code()

    def instantiation(self, prefix: str) -> Code:
        yield from _BufferedComponentInstantiation(prefix=prefix).code()


@dataclasses.dataclass
class _SignalsForBufferlessComponent:
    prefix: str
    data_width: int

    def _logic_vector(self, suffix, width) -> str:
        return (
            f"signal {self.prefix}_{suffix} : std_logic_vector({width - 1} downto 0);"
        )

    def _std_logic(self, suffix) -> str:
        return f"signal {self.prefix}_{suffix} : std_logic := '0';"

    def code(self) -> Code:
        code = [self._std_logic(suffix) for suffix in ("enable", "clock")]
        code.extend(
            [
                self._logic_vector(suffix, width)
                for suffix, width in (
                    ("x", self.data_width),
                    ("y", self.data_width),
                )
            ]
        )

        return code


@dataclasses.dataclass
class _SignalsForComponentWithBuffer(_SignalsForBufferlessComponent):
    prefix: str
    data_width: int
    x_address_width: int
    y_address_width: int

    def code(self) -> Code:
        code = list(super().code())
        code.append(self._std_logic("done"))
        code.extend(
            [
                self._logic_vector(suffix, width)
                for suffix, width in (
                    ("x_address", self.x_address_width),
                    ("y_address", self.y_address_width),
                )
            ]
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
