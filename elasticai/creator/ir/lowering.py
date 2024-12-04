from abc import abstractmethod
from collections.abc import Callable, Iterable
from functools import wraps
from typing import Generic, Protocol, TypeVar, overload

from .function_registry import FunctionRegistry as _Registry


class Lowerable(Protocol):
    @property
    @abstractmethod
    def type(self) -> str: ...


Tin = TypeVar("Tin", bound="Lowerable")
Tout = TypeVar("Tout")


class LoweringPass(Generic[Tin, Tout]):
    def __init__(self) -> None:
        def key_lookup_fn(x: Tin) -> str:
            return x.type

        self._fns: _Registry[Tin, Iterable[Tout]] = _Registry(key_lookup_fn)

    @overload
    def register(self, fn: Callable[[Tin], Tout], /) -> Callable[[Tin], Tout]: ...

    @overload
    def register(
        self, name: str, /
    ) -> Callable[[Callable[[Tin], Tout]], Callable[[Tin], Tout]]: ...

    def register(self, arg, /):
        self._check_for_redefinition(arg)

        if isinstance(arg, str):

            def _reg(fn):
                wrapper = self._create_iterfn_wrapper(fn)
                self._fns.register(arg)(wrapper)
                return fn

            return _reg

        self._fns.register(self._create_iterfn_wrapper(arg))
        return arg

    def _create_iterfn_wrapper(self, fn):
        @wraps(fn)
        def wrapper(arg):
            return (fn(arg),)

        return wrapper

    def __call__(self, args: Iterable[Tin]) -> Iterable[Tout]:
        for arg in args:
            yield from self._fns.call(arg)

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
        return self._fns.register(arg)

    def _check_for_redefinition(self, arg):
        if arg in self._fns:
            raise ValueError(f"function for {arg} already defined in lowering pass")
