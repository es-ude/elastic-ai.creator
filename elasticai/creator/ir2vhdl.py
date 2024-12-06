from collections import namedtuple
from collections.abc import Callable, Iterable
from typing import overload

from elasticai.creator.ir import Graph, Lowerable, LoweringPass, Node
from elasticai.creator.plugin_loader import PluginLoader


class VhdlNode(Node):
    entity: str


class VhdlEntityIr(Graph, Lowerable):
    def __init__(self, name: str, type: str) -> None:
        super().__init__()
        self._name = name
        self._type = type

    @property
    def name(self) -> str:
        return self._name

    @property
    def type(self) -> str:
        return self._type


def process_vhdl_template(
    template: Iterable[str],
) -> Callable[[VhdlEntityIr], Iterable[str]]:
    def processor(entity: VhdlEntityIr) -> Iterable[str]:
        yield ""

    return processor


SourceFile = namedtuple("SourceFile", ("name", "code"))


class Vhdl2Ir(LoweringPass[VhdlEntityIr, tuple[str, Iterable[str]]]):
    pass


class _GeneratedCodeType[Tin: Lowerable, Tout]:
    def __init__(self, name: str, fn: Callable[[Tin], Tout]) -> None:
        self._fn = fn
        self._name = name

    def __call__(self, arg: Tin, /) -> Tout:
        return self._fn(arg)

    def load(self, lower: PluginLoader[Tin, Tout]) -> None:
        lower.register(self._name, self._fn)


class _GeneratedIterableCodeType[Tin: Lowerable, Tout]:
    def __init__(self, name: str, fn: Callable[[Tin], Iterable[Tout]]) -> None:
        self._fn = fn
        self._name = name

    def __call__(self, arg: Tin, /) -> Iterable[Tout]:
        return self._fn(arg)

    def load(self, lower: PluginLoader[Tin, Tout]) -> None:
        lower.register_iterable(self._name, self._fn)


@overload
def type_handler[
    Tin: Lowerable, Tout
](arg: str, /,) -> Callable[[Callable[[Tin], Tout]], _GeneratedCodeType[Tin, Tout]]: ...


@overload
def type_handler[
    Tin: Lowerable, Tout
](arg: Callable[[Tin], Tout], /,) -> _GeneratedCodeType[Tin, Tout]: ...


def type_handler[
    Tin: Lowerable, Tout
](arg: str | Callable[[Tin], Tout], /) -> (
    Callable[[Callable[[Tin], Tout]], _GeneratedCodeType[Tin, Tout]]
    | _GeneratedCodeType[Tin, Tout]
):
    if not isinstance(arg, str):
        return _GeneratedCodeType(arg.__name__, arg)

    def wrapper(fn: Callable[[Tin], Tout]) -> _GeneratedCodeType[Tin, Tout]:
        return _GeneratedCodeType(arg, fn)

    return wrapper


@overload
def iterable_type_handler[
    Tin: Lowerable, Tout
](
    arg: str,
    /,
) -> Callable[
    [Callable[[Tin], Iterable[Tout]]], _GeneratedIterableCodeType[Tin, Tout]
]: ...


@overload
def iterable_type_handler[
    Tin: Lowerable, Tout
](
    arg: Callable[[Tin], Iterable[Tout]],
    /,
) -> _GeneratedIterableCodeType[
    Tin, Tout
]: ...


def iterable_type_handler[
    Tin: Lowerable, Tout
](arg: str | Callable[[Tin], Iterable[Tout]], /) -> (
    Callable[
        [Callable[[Tin], Iterable[Tout]]],
        _GeneratedIterableCodeType[Tin, Tout],
    ]
    | _GeneratedIterableCodeType[Tin, Tout]
):
    if not isinstance(arg, str):
        return _GeneratedIterableCodeType(arg.__name__, arg)

    def wrapper(
        fn: Callable[[Tin], Iterable[Tout]],
    ) -> _GeneratedIterableCodeType[Tin, Tout]:
        return _GeneratedIterableCodeType(arg, fn)

    return wrapper
