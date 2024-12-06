from collections.abc import Callable, Iterable
from typing import Protocol, overload

from elasticai.creator.ir import Lowerable
from elasticai.creator.plugin import PluginLoader


class GeneratedCodeType[Tin: Lowerable, Tout]:
    def __init__(self, name: str, fn: Callable[[Tin], Tout]) -> None:
        self._fn = fn
        self._name = name

    def __call__(self, arg: Tin, /) -> Tout:
        return self._fn(arg)

    def load(self, lower: PluginLoader[Tin, Tout]) -> None:
        lower.register(self._name, self._fn)


class GeneratedIterableCodeType[Tin: Lowerable, Tout]:
    def __init__(self, name: str, fn: Callable[[Tin], Iterable[Tout]]) -> None:
        self._fn = fn
        self._name = name

    def __call__(self, arg: Tin, /) -> Iterable[Tout]:
        return self._fn(arg)

    def load(self, lower: PluginLoader[Tin, Tout]) -> None:
        lower.register_iterable(self._name, self._fn)


@overload
def code_type[
    Tin: Lowerable, Tout
](arg: str, /,) -> Callable[[Callable[[Tin], Tout]], GeneratedCodeType[Tin, Tout]]: ...


@overload
def code_type[
    Tin: Lowerable, Tout
](arg: Callable[[Tin], Tout], /,) -> GeneratedCodeType[Tin, Tout]: ...


def code_type[
    Tin: Lowerable, Tout
](arg: str | Callable[[Tin], Tout], /) -> (
    Callable[[Callable[[Tin], Tout]], GeneratedCodeType[Tin, Tout]]
    | GeneratedCodeType[Tin, Tout]
):
    if not isinstance(arg, str):
        return GeneratedCodeType(arg.__name__, arg)

    def wrapper(fn: Callable[[Tin], Tout]) -> GeneratedCodeType[Tin, Tout]:
        return GeneratedCodeType(arg, fn)

    return wrapper


@overload
def iterable_code_type[
    Tin: Lowerable, Tout
](
    arg: str,
    /,
) -> Callable[
    [Callable[[Tin], Iterable[Tout]]], GeneratedIterableCodeType[Tin, Tout]
]: ...


@overload
def iterable_code_type[
    Tin: Lowerable, Tout
](arg: Callable[[Tin], Iterable[Tout]], /,) -> GeneratedIterableCodeType[Tin, Tout]: ...


def iterable_code_type[
    Tin: Lowerable, Tout
](arg: str | Callable[[Tin], Iterable[Tout]], /) -> (
    Callable[
        [Callable[[Tin], Iterable[Tout]]],
        GeneratedIterableCodeType[Tin, Tout],
    ]
    | GeneratedIterableCodeType[Tin, Tout]
):
    if not isinstance(arg, str):
        return GeneratedIterableCodeType(arg.__name__, arg)

    def wrapper(
        fn: Callable[[Tin], Iterable[Tout]],
    ) -> GeneratedIterableCodeType[Tin, Tout]:
        return GeneratedIterableCodeType(arg, fn)

    return wrapper


class DummyLowerable(Lowerable, Protocol):
    more_data: list[str]


@code_type
def convolution(x: DummyLowerable) -> str:
    return "_".join(x.more_data)
