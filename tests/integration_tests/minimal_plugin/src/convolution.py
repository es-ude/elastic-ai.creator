from collections.abc import Callable
from typing import Protocol

from elasticai.creator.ir import Lowerable, LoweringPass
from elasticai.creator.plugin import PluginSymbol


class DummyLowerable(Lowerable, Protocol):
    more_data: list[str]


class CallablePluginSymbol(PluginSymbol):
    def __init__(self, fn: Callable[[DummyLowerable], str]):
        self._fn = fn

    def load_into(self, receiver: LoweringPass) -> None:
        receiver.register(self._fn.__name__)(self._fn)

    def __call__(self, x: DummyLowerable) -> str:
        return self._fn(x)


def make_plugin_symbol(fn: Callable[[DummyLowerable], str]) -> PluginSymbol:
    return CallablePluginSymbol(fn)


@make_plugin_symbol
def convolution(x: DummyLowerable) -> str:
    return "_".join(x.more_data)
