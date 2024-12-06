from collections.abc import Callable, Iterable
from typing import overload

from pytest import fixture

from elasticai.creator.ir import Lowerable, LoweringPass
from elasticai.creator.plugin import Plugin, PluginLoader


@fixture
def plugin() -> Plugin:
    return Plugin(
        name="plugin_test",
        target_platform="platform",
        target_runtime="runtime",
        version="0.1",
        api_version="0.1",
        generated=("my_type",),
        templates=tuple(),
        static_files=tuple(),
        package="tests.unit",
    )


def register(fn):
    pass


def register_iterable(fn):
    pass


class DummyLowerable(Lowerable):
    def __init__(self, type: str, more_data: list[str]):
        self._type = type
        self.more_data: list[str] = more_data

    @property
    def type(self) -> str:
        return self._type


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


@code_type
def my_type(d: DummyLowerable) -> str:
    return "_".join(d.more_data)


@fixture
def make_lowerable(plugin) -> Callable[[list[str]], DummyLowerable]:
    def dummy(more_data: list[str]) -> DummyLowerable:
        return DummyLowerable(plugin.generated[0], more_data)

    return dummy


def test_can_load_and_run_lowering_handler(
    make_lowerable: Callable[[list[str]], DummyLowerable],
) -> None:
    lower: LoweringPass[DummyLowerable, str] = LoweringPass()
    loader = PluginLoader(lower)
    lowerable = make_lowerable(["some", "important", "information"])
    loader.load_generated(my_type)
    assert ("some_important_information",) == tuple(lower([lowerable]))


def test_can_load_entire_plugin(
    plugin: Plugin, make_lowerable: Callable[[list[str]], DummyLowerable]
) -> None:
    lower: LoweringPass[DummyLowerable, str] = LoweringPass()
    loader = PluginLoader(lower)
    lowerable = make_lowerable(["some", "important", "information"])
    loader.load(plugin)
    assert ("some_more_information",) == tuple(lower((lowerable,)))
