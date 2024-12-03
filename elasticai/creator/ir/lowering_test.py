from types import SimpleNamespace

from .function_registry import FunctionRegistry

"""
Tests:
    - register a function by function name
    - register a function by custom name
    - call fns and chain results, dispatching arguments by type

For later in LoweringPass:
    - convert function into one that returns iterable in case it is not already
"""


def test_calls_a_registered_function():
    r = FunctionRegistry(dispatch_key_fn=lambda x: x.type)
    c = SimpleNamespace(name="c0", type="convolution")

    @r.register
    def convolution(x) -> str:
        return f"conv{x.name}"

    assert "convc0" == r.call(c)


def test_calling_a_function_with_custom_name():
    r = FunctionRegistry(dispatch_key_fn=lambda x: x.type)
    c = SimpleNamespace(name="c0", type="convolution")

    @r.register("convolution")
    def fn(x) -> str:
        return f"conv{x.name}"

    assert "convc0" == r.call(c)
