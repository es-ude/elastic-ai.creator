import warnings
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from functools import wraps
from importlib import resources as res
from typing import Any, Protocol, TypeAlias, cast, overload

import elasticai.creator.function_dispatch as F
from elasticai.creator import graph as g
from elasticai.creator import ir
from elasticai.creator import plugin as pl
from elasticai.creator.ir import Edge as Edge


class Node(ir.Node):
    implementation: str


Code: TypeAlias = tuple[str, Sequence[str]]


@dataclass
class PluginSpec(pl.PluginSpec):
    generated: tuple[str, ...]
    static_files: tuple[str, ...]


class Implementation(ir.Implementation[Node, ir.Edge]):
    type: str

    def __init__(self, *, graph: g.Graph[str], data: dict[str, ir.Attribute]):
        super().__init__(graph=graph, data=data)


class Ir2Verilog(ir.LoweringPass[Implementation, Code]):
    def __init__(self) -> None:
        super().__init__()
        self.__static_files: dict[str, Callable[[], str]] = {}
        self._loader = PluginLoader(self)

    def register_static(self, name: str, fn: Callable[[], str]) -> None:
        self.__static_files[name] = fn

    def __call__(self, args: Iterable[Implementation]) -> Iterator[Code]:
        for name, content in super().__call__(args):
            yield f"{name}.v", content
        for name, fn in self.__static_files.items():
            yield name, fn()

    def load_from_package(self, package: str) -> None:
        self._loader.load_from_package(package)


class PluginSymbol(pl.PluginSymbol[Ir2Verilog], Protocol):
    pass


class PluginLoader(pl.PluginLoader[Ir2Verilog]):
    """PluginLoader for Ir2Verilog passes."""

    def __init__(self, lowering: Ir2Verilog):
        builder: pl.SymbolFetcherBuilder[PluginSpec, Ir2Verilog] = (
            pl.SymbolFetcherBuilder(PluginSpec)
        )
        fetcher: pl.SymbolFetcher[Ir2Verilog] = (
            builder.add_fn(self.__get_generated)
            .add_fn(_StaticFile.make_symbols)
            .build()
        )
        super().__init__(
            fetch=fetcher,
            plugin_receiver=lowering,
        )

    def load_from_package(self, package: str) -> None:
        if "." not in package:
            package = f"elasticai.creator_plugins.{package}"
        super().load_from_package(package)

    @staticmethod
    def __get_generated(plugin: PluginSpec) -> Iterator[PluginSymbol]:
        if plugin.target_runtime == "verilog":
            yield from pl.import_symbols(plugin.package, plugin.generated)


class _StaticFile(PluginSymbol):
    _subfolder = "verilog"

    def __init__(self, name: str, package: str):
        self._name = name
        self._package = package

    @property
    def name(self) -> str:
        return self._name

    def load_into(self, receiver: Ir2Verilog):
        receiver.register_static(self.name, self)

    @classmethod
    def make_symbols(cls, p: PluginSpec) -> Iterator[pl.PluginSymbol[Ir2Verilog]]:
        if p.target_runtime == cls._subfolder:
            for name in p.static_files:
                yield cls(name=name, package=p.package)

    def __call__(self) -> str:
        file = res.files(self._package).joinpath(f"{self._subfolder}/{self.name}")
        return file.read_text()


TypeHandlerFn: TypeAlias = Callable[[Implementation], Code]


class _LegacyRegistrar[C: Code | Iterable[Code]](Protocol):
    @overload
    def __call__(self) -> "_LegacyRegistrar[C]": ...
    @overload
    def __call__(self, fn: Callable[[Implementation], C], /) -> PluginSymbol: ...
    @overload
    def __call__(
        self, name: str | None, fn: Callable[[Implementation], C], /
    ) -> PluginSymbol: ...

    def __call__(self, *args) -> Any: ...


def _legacy_registrar[C: Iterable[Code] | Code](
    handler_creator: Callable[
        [str | None, Callable[[Implementation], C]], PluginSymbol
    ],
) -> _LegacyRegistrar[C]:
    registrar = F.registrar(handler_creator)

    @wraps(handler_creator)
    def wrapper(*args):
        if len(args) == 1 and callable(args[0]):
            warnings.warn(
                "You're using a type handler as `@type_handler` this is deprecated and will be removed in the future. Use `@type_handler()` instead.",
                category=DeprecationWarning,
                stacklevel=2,
            )
            return handler_creator(
                args[0].__name__, cast(Callable[[Implementation], C], args[0])
            )
        return registrar(*args)

    return cast(_LegacyRegistrar, wrapper)


def _check_and_get_fn_name(name: str | None, fn: Callable) -> str:
    if name is None:
        if hasattr(fn, "__name__") and isinstance(fn.__name__, str):
            name = fn.__name__
        else:
            raise Exception(
                "provided type handler has to be a function or named explicitly"
            )
    return name


@_legacy_registrar
def type_handler(
    name: str | None, fn: Callable[[Implementation], Code]
) -> pl.PluginSymbolFn[Ir2Verilog, [Implementation], Code]:
    name = _check_and_get_fn_name(name, fn)

    def load_into(lower: Ir2Verilog) -> None:
        lower.register(name)(fn)  # ty: ignore

    return pl.make_plugin_symbol(load_into, fn)


@_legacy_registrar
def type_handler_iterable(
    name: str | None, fn: Callable[[Implementation], Iterable[Code]]
) -> pl.PluginSymbolFn[Ir2Verilog, [Implementation], Iterable[Code]]:
    name = _check_and_get_fn_name(name, fn)

    def load_into(lower: Ir2Verilog) -> None:
        lower.register_iterable(name)(fn)  # ty: ignore

    return pl.make_plugin_symbol(load_into, fn)
