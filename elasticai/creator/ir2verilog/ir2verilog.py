from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from importlib import resources as res
from itertools import starmap
from typing import TypeAlias

import elasticai.creator.function_dispatch as FD
from elasticai.creator import plugin as pl
from elasticai.creator.hdl_ir import (
    DataGraph,
    IrFactory,
    NonIterableTypeHandler,
    Registry,
    TypeHandler,
    _check_and_get_name_fn,
)
from elasticai.creator.ir import ir_v2 as ir

factory = IrFactory()

Code: TypeAlias = tuple[str, Sequence[str]]


@dataclass
class PluginSpec(pl.PluginSpec):
    generated: tuple[str, ...]
    static_files: tuple[str, ...]


class Ir2Verilog:
    def __init__(self) -> None:
        self.__static_files: dict[str, Callable[[], str]] = {}

    def __call__(
        self, root: DataGraph, registry: Registry, default_root_name="root"
    ) -> Iterable[Code]:
        registry = self._give_names_to_registry_graphs(registry)
        if "name" not in root.attributes:
            root = root.with_attributes(root.attributes | dict(name=default_root_name))
        yield from self._handle_type(root, registry)
        for g in registry.values():
            for name, code in self._handle_type(g, registry):
                yield f"{name}.v", code
        for name, fn in self.__static_files.items():
            yield name, fn()

    def _give_names_to_registry_graphs(self, registry: Registry) -> Registry:
        def give_name(name: str, g: DataGraph) -> tuple[str, DataGraph]:
            if "name" not in g.attributes:
                return name, g.with_attributes(g.attributes | dict(name=name))
            return name, g

        return ir.Registry(starmap(give_name, registry.items()))

    @FD.dispatch_method(str)
    def _handle_type(
        self, fn: TypeHandler, graph: DataGraph, registry: Registry
    ) -> Iterable[Code]:
        return fn(graph, registry)

    @_handle_type.key_from_args
    def _get_key(self, graph: DataGraph, registry: Registry) -> str:
        return graph.type

    @staticmethod
    def _check_and_get_name(name: str | None, fn: Callable) -> str:
        return _check_and_get_name_fn(name, fn)

    @FD.registrar_method
    def register_static(
        self, name: str | None, fn: Callable[[], str]
    ) -> Callable[[], str]:
        self.__static_files[self._check_and_get_name(name, fn)] = fn
        return fn

    @FD.registrar_method
    def register(self, name: str | None, fn: TypeHandler) -> TypeHandler:
        name = self._check_and_get_name(name, fn)
        self._handle_type.register(name, fn)
        return fn

    @FD.registrar_method
    def override(self, name: str | None, fn: TypeHandler) -> TypeHandler:
        name = self._check_and_get_name(name, fn)
        self._handle_type.override(name, fn)
        return fn


type PluginSymbol = pl.PluginSymbolFn[Ir2Verilog, [DataGraph, Registry], Code]
type PluginSymbolIter = pl.PluginSymbolFn[
    Ir2Verilog, [DataGraph, Registry], Iterable[Code]
]


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
    def __get_generated(plugin: PluginSpec) -> Iterator[pl.PluginSymbol]:
        if plugin.target_runtime == "verilog":
            yield from pl.import_symbols(plugin.package, plugin.generated)


class _StaticFile(pl.PluginSymbol[Ir2Verilog]):
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


@FD.registrar
def type_handler(name: str | None, fn: NonIterableTypeHandler) -> PluginSymbol:
    name = _check_and_get_name_fn(name, fn)

    def load_into(lower: Ir2Verilog) -> None:
        def wrapper(*args, **kwargs):
            yield fn(*args, **kwargs)

        lower.register(name, wrapper)

    return pl.make_plugin_symbol(load_into, fn)


@FD.registrar
def type_handler_iterable(name: str | None, fn: TypeHandler) -> PluginSymbolIter:
    name = _check_and_get_name_fn(name, fn)
    if name is None:
        name = fn.__name__

    def load_into(lower: Ir2Verilog) -> None:
        lower.register(name, fn)

    return pl.make_plugin_symbol(load_into, fn)
