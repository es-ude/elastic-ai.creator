from collections.abc import Callable
from typing import ParamSpec, TypeVar

from elasticai.creator.ir.function_registry import FnRegisterer as Register
from elasticai.creator.ir.function_registry import return_as_iterable

P = ParamSpec("P")
T = TypeVar("T")
Fn = TypeVar("Fn", bound=Callable)


def test_convert_callable_to_return_iterable() -> None:
    @return_as_iterable
    def fn(x: str) -> str:
        return x

    assert ("hello",) == tuple(fn("hello"))


def test_can_register() -> None:
    registry = {}

    def callback(arg: str, fn: Callable[[str], str]) -> None:
        registry[arg] = fn

    register = Register(callback)

    @register
    def fn(x: str) -> str:
        return f"fn {x}"

    assert "fn 5" == fn("5")
    assert registry["fn"] == fn
