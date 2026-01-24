from collections.abc import Callable
from typing import Protocol


class DummyLowerable(Protocol):
    more_data: list[str]


class Translator(Protocol):
    def register(self, name: str, fn: Callable) -> Callable: ...


class CallablePluginSymbol:
    def __init__(self, fn: Callable[[DummyLowerable], str]):
        self._fn = fn

    def load_vhdl(self, receiver: Translator) -> None:
        receiver.register(self._fn.__name__, self._fn)

    def load_minimal(self, receiver: Translator) -> None:
        receiver.register(self._fn.__name__, self._fn)

    def load_other(self, receiver: Translator) -> None:
        receiver.register("other_" + self._fn.__name__, self._fn)

    def __call__(self, x: DummyLowerable) -> str:
        return self._fn(x)


def make_plugin_symbol(fn: Callable[[DummyLowerable], str]) -> CallablePluginSymbol:
    return CallablePluginSymbol(fn)


@make_plugin_symbol
def convolution(x: DummyLowerable) -> str:
    return "_".join(x.more_data)
