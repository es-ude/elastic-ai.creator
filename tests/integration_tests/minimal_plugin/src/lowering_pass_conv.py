from collections.abc import Callable
from typing import Protocol

from .convolution import DummyLowerable


class Translator(Protocol):
    def register(self, name: str) -> Callable: ...


class LoweringPassPluginSymbol:
    def __init__(self, fn: Callable[[DummyLowerable], str]):
        self._fn = fn

    def load_lowering_pass_test(self, receiver: Translator) -> None:
        receiver.register(self._fn.__name__)(self._fn)

    def __call__(self, x: DummyLowerable) -> str:
        return self._fn(x)


def make_plugin_symbol(
    fn: Callable[[DummyLowerable], str],
) -> LoweringPassPluginSymbol:
    return LoweringPassPluginSymbol(fn)


@make_plugin_symbol
def lowering_pass_conv(x: DummyLowerable) -> str:
    return "_".join(x.more_data)
