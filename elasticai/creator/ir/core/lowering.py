import warnings
from abc import abstractmethod
from collections.abc import Callable, Iterable, Iterator
from functools import wraps
from typing import Generic, ParamSpec, Protocol, TypeVar

import elasticai.creator.function_dispatch as F


class Lowerable(Protocol):
    @property
    @abstractmethod
    def type(self) -> str: ...


Tin = TypeVar("Tin", bound="Lowerable")
Tout = TypeVar("Tout")


class LoweringPass(Generic[Tin, Tout]):
    _dispatcher: F.KeyedDispatcherDescriptor[
        [Tin],
        [Tin],
        Iterable[Tout],
        Iterable[Tout],
        "LoweringPass",
        str,
    ] = F.KeyedDispatcherDescriptor()

    @_dispatcher.key_from_args
    def _key_from_args(self, x: Tin) -> str:
        return x.type

    @F.registrar_method
    def register(
        self, name: str | None, fn: Callable[[Tin], Tout]
    ) -> Callable[[Tin], Tout]:
        if name is None:
            name = fn.__name__
        wrapper = return_as_iterable(fn)
        self._dispatcher.register(name, wrapper)
        return fn

    @F.registrar_method
    def register_override(
        self, name: str | None, fn: Callable[[Tin], Tout]
    ) -> Callable[[Tin], Tout]:
        if name is None:
            name = fn.__name__
        wrapper = return_as_iterable(fn)
        self._dispatcher.override(name, wrapper)
        return fn

    @F.registrar_method
    def register_iterable(
        self, name: str | None, fn: Callable[[Tin], Iterable[Tout]]
    ) -> Callable[[Tin], Iterable[Tout]]:
        if name is None:
            name = fn.__name__
        self._dispatcher.register(name, fn)
        return fn

    @F.registrar_method
    def register_iterable_override(
        self, name: str | None, fn: Callable[[Tin], Iterable[Tout]]
    ) -> Callable[[Tin], Iterable[Tout]]:
        if name is None:
            name = fn.__name__
        self._dispatcher.override(name, fn)
        return fn

    @_dispatcher.dispatch_for
    def _run(self, fn, x: Tin) -> Iterator[Tout]:
        yield from fn(x)

    def __call__(self, args: Iterable[Tin]) -> Iterator[Tout]:
        for arg in args:
            yield from self._run(arg)


P = ParamSpec("P")


def return_as_iterable(fn: Callable[P, Tout]) -> Callable[P, Iterator[Tout]]:
    @wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> Iterator[Tout]:
        yield fn(*args, **kwargs)

    return wrapper
