from abc import abstractmethod
from collections.abc import Callable, Iterable, Iterator
from functools import wraps
from typing import Protocol

import elasticai.creator.function_dispatch as F


class Lowerable(Protocol):
    @property
    @abstractmethod
    def type(self) -> str: ...


class LoweringPass[Tin: Lowerable, Tout]:
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

    def _check_and_get_name(self, name: str | None, fn: Callable) -> str:
        if name is None:
            if hasattr(fn, "__name__") and isinstance(fn.__name__, str):
                name = fn.__name__
            else:
                raise TypeError(f"You have to explicitly provide a name for {type(fn)}")
        return name

    @F.registrar_method  # ty: ignore
    def register(
        self, name: str | None, fn: Callable[[Tin], Tout], /
    ) -> Callable[[Tin], Tout]:
        name = self._check_and_get_name(name, fn)
        wrapper = return_as_iterable(fn)
        self._dispatcher.register(name, wrapper)
        return fn

    @F.registrar_method  # ty: ignore
    def register_override(
        self, name: str | None, fn: Callable[[Tin], Tout]
    ) -> Callable[[Tin], Tout]:
        name = self._check_and_get_name(name, fn)
        wrapper = return_as_iterable(fn)
        self._dispatcher.override(name, wrapper)
        return fn

    @F.registrar_method  # ty: ignore
    def register_iterable(
        self, name: str | None, fn: Callable[[Tin], Iterable[Tout]]
    ) -> Callable[[Tin], Iterable[Tout]]:
        name = self._check_and_get_name(name, fn)
        self._dispatcher.register(name, fn)
        return fn

    @F.registrar_method  # ty: ignore
    def register_iterable_override(
        self, name: str | None, fn: Callable[[Tin], Iterable[Tout]]
    ) -> Callable[[Tin], Iterable[Tout]]:
        name = self._check_and_get_name(name, fn)
        self._dispatcher.override(name, fn)
        return fn

    @_dispatcher.dispatch_for
    def _run(self, fn, x: Tin) -> Iterator[Tout]:
        yield from fn(x)

    def __call__(self, args: Iterable[Tin]) -> Iterator[Tout]:
        for arg in args:
            yield from self._run(arg)


def return_as_iterable[Tout, **P](fn: Callable[P, Tout]) -> Callable[P, Iterator[Tout]]:
    @wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> Iterator[Tout]:
        yield fn(*args, **kwargs)

    return wrapper
