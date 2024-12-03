from typing import Callable, Generic, TypeVar, overload

Tin = TypeVar("Tin")
Tout = TypeVar("Tout")


class FunctionRegistry(Generic[Tin, Tout]):
    """
    Functions can be registered with custom key or by
    using the functions name as key.
    Upon `call` the registry will use the function
    specified by the `dispatch_key_fn` and the argument
    to `call` to process the argument and return the result.
    """

    def __init__(self, dispatch_key_fn: Callable[[Tin], str]) -> None:
        self._key_fn = dispatch_key_fn
        self._fns: dict[str, Callable[[Tin], Tout]] = dict()

    @overload
    def register(self, fn: Callable[[Tin], Tout], /) -> Callable[[Tin], Tout]: ...

    @overload
    def register(
        self, name: str, /
    ) -> Callable[[Callable[[Tin], Tout]], Callable[[Tin], Tout]]: ...

    def register(self, arg, /):
        def _make_register_fn(key):
            def _register_fn(fn):
                self._fns[key] = fn
                return fn

            return _register_fn

        if isinstance(arg, str):
            return _make_register_fn(arg)

        return _make_register_fn(arg.__name__)(arg)

    def call(self, arg: Tin) -> Tout:
        return self._fns[self._key_fn(arg)](arg)
