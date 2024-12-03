from collections.abc import Callable, Iterable
from typing import Generic, Protocol, TypeVar, overload

from .function_registry import FunctionRegistry as _Registry


class Lowerable(Protocol):
    @property
    def type(self) -> str: ...


Tin = TypeVar("Tin", bound="Lowerable")
Tout = TypeVar("Tout")


class LoweringPass(Generic[Tin, Tout]):
    def __init__(self) -> None:
        def key_lookup_fn(x: Tin) -> str:
            return x.type

        self._fns: _Registry[Tin, Tout] = _Registry(key_lookup_fn)
        self._iterfns: _Registry[Tin, Iterable[Tout]] = _Registry(key_lookup_fn)

    @overload
    def register(self, fn: Callable[[Tin], Tout], /) -> Callable[[Tin], Tout]: ...

    @overload
    def register(
        self, name: str, /
    ) -> Callable[[Callable[[Tin], Tout]], Callable[[Tin], Tout]]: ...

    def _check_for_redefinition(self, arg):
        if arg in self._fns or arg in self._iterfns:
            raise ValueError(f"function for {arg} already defined in lowering pass")

    def register(self, arg, /):
        self._check_for_redefinition(arg)
        return self._fns.register(arg)

    def __call__(self, args: Iterable[Tin]) -> Iterable[Tout]:
        for arg in args:
            if self._fns.can_dispatch(arg):
                yield self._fns.call(arg)
            else:
                yield from self._iterfns.call(arg)

    @overload
    def register_iterable(
        self, name: str, /
    ) -> Callable[
        [Callable[[Tin], Iterable[Tout]]], Callable[[Tin], Iterable[Tout]]
    ]: ...

    @overload
    def register_iterable(
        self, fn: Callable[[Tin], Iterable[Tout]], /
    ) -> Callable[[Tin], Iterable[Tout]]: ...

    def register_iterable(self, arg, /):
        self._check_for_redefinition(arg)
        return self._iterfns.register(arg)
