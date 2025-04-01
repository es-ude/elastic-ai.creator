from types import SimpleNamespace

import pytest

from elasticai.creator.function_utils import KeyedFunctionDispatcher
from elasticai.creator.ir import LoweringPass

"""
Tests:

For later in LoweringPass:
    - convert function into one that returns iterable in case it is not already
"""


def test_calls_a_registered_function():
    r = KeyedFunctionDispatcher(dispatch_key_fn=lambda x: x.type)
    c = SimpleNamespace(name="c0", type="convolution")

    @r.register
    def convolution(x) -> str:
        return f"conv{x.name}"

    assert "convc0" == r.call(c)


def test_calling_a_function_with_custom_name():
    r = KeyedFunctionDispatcher(dispatch_key_fn=lambda x: x.type)
    c = SimpleNamespace(name="c0", type="convolution")

    @r.register("convolution")
    def fn(x) -> str:
        return f"conv{x.name}"

    assert "convc0" == r.call(c)


def test_register_lowering_fn_with_fn_name() -> None:
    c = SimpleNamespace(name="c0", type="convolution")
    p: LoweringPass[SimpleNamespace, str] = LoweringPass()

    @p.register
    def convolution(x: SimpleNamespace) -> str:
        return f"conv{x.name}"

    assert ("convc0",) == tuple(p((c,)))


def test_can_directly_call_fn_as_defined() -> None:
    c = SimpleNamespace(name="c0", type="convolution")
    p: LoweringPass[SimpleNamespace, str] = LoweringPass()

    @p.register
    def convolution(x: SimpleNamespace) -> str:
        return f"conv{x.name}"

    assert "convc0" == convolution(c)


def test_register_lowering_fn_with_custom_name() -> None:
    c = SimpleNamespace(name="c0", type="convolution")
    p: LoweringPass[SimpleNamespace, str] = LoweringPass()

    @p.register("convolution")
    def f(x: SimpleNamespace) -> str:
        return f"conv{x.name}"

    assert ("convc0",) == tuple(p((c,)))


def test_lowering_iterates_over_input() -> None:
    c = SimpleNamespace(name="c0", type="convolution")
    d = SimpleNamespace(name="c1", type="convolution")
    p: LoweringPass[SimpleNamespace, str] = LoweringPass()

    @p.register
    def convolution(x: SimpleNamespace) -> str:
        return f"conv{x.name}"

    assert {"convc0", "convc1"} == set(p((c, d)))


def test_can_register_and_call_iter_fn() -> None:
    c = SimpleNamespace(name="c0", type="convolution")
    p: LoweringPass[SimpleNamespace, str] = LoweringPass()

    @p.register_iterable
    def convolution(x: SimpleNamespace) -> tuple[str, ...]:
        return f"conv{x.name}", "more content"

    assert {"convc0", "more content"} == set(p((c,)))


def test_registering_a_name_as_iterable_and_non_iterable_yields_error() -> None:
    p: LoweringPass = LoweringPass()

    @p.register
    def a(x):
        return x

    with pytest.raises(
        ValueError, match="function for a already defined in lowering pass"
    ):

        @p.register_iterable("a")
        def b(x):
            return (x,)
