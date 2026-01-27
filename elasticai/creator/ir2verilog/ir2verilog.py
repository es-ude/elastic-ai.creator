import warnings
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from importlib import resources as res
from itertools import starmap
from typing import Any, Protocol, TypeAlias, override

import elasticai.creator.function_dispatch as FD
from elasticai.creator import ir
from elasticai.creator import plugin as _pl
from elasticai.creator.hdl_ir import (
    DataGraph,
    IrFactory,
    NonIterableTypeHandler,
    Registry,
    TypeHandler,
    _check_and_get_name_fn,
)
from elasticai.creator.plugin import PluginLoaderBase, StaticFileBase

factory = IrFactory()

Code: TypeAlias = tuple[str, Sequence[str]]


@dataclass
class PluginSpec(_pl.PluginSpec):
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


class PluginSymbol(Protocol):
    def load_verilog(self, receiver: Ir2Verilog) -> None: ...


class _StaticFileSymbol:
    def __init__(self, file: StaticFileBase):
        self._file = file

    def load_verilog(self, receiver: Ir2Verilog) -> None:
        receiver.register_static(self._file.name, self._file.get_content)


class PluginLoader(PluginLoaderBase):
    """PluginLoader for Ir2Verilog passes."""

    def __init__(self, lowering: Ir2Verilog):
        self._receiver = lowering
        super().__init__(PluginSpec)

    @override
    def filter_plugin_dicts(
        self, plugins: Iterable[dict[str, Any]]
    ) -> Iterable[dict[str, Any]]:
        for p in plugins:
            if p["target_runtime"] == "verilog":
                yield p

    @override
    def load_symbol(self, symbol: PluginSymbol) -> None:
        if hasattr(symbol, "load_verilog"):
            symbol.load_verilog(self._receiver)
        elif hasattr(symbol, "load_into"):
            warnings.warn(
                "Loading legacy plugin symbol, this behaviour will be removed in the future ensure your plugin symbol provides a load_verilog method",
                stacklevel=2,
                category=DeprecationWarning,
            )
            symbol.load_into(self._receiver)
        else:
            raise TypeError("Failed to load plugin symbol")

    @override
    def get_symbols(self, specs: Iterable[PluginSpec]) -> Iterable[PluginSymbol]:
        for spec in specs:
            yield from _pl.import_symbols(spec.package, spec.generated)
            for static_name in spec.static_files:
                yield _StaticFileSymbol(
                    _pl.StaticFileBase(
                        name=static_name, package=spec.package, subfolder="verilog"
                    )
                )


class _StaticFile:
    _subfolder = "verilog"

    def __init__(self, name: str, package: str):
        self._name = name
        self._package = package

    @property
    def name(self) -> str:
        return self._name

    def load_verilog(self, receiver: Ir2Verilog):
        receiver.register_static(self.name, self)

    @classmethod
    def make_symbols(cls, p: PluginSpec) -> Iterator[PluginSymbol]:
        if p.target_runtime == cls._subfolder:
            for name in p.static_files:
                yield cls(name=name, package=p.package)

    def __call__(self) -> str:
        file = res.files(self._package).joinpath(f"{self._subfolder}/{self.name}")
        return file.read_text()


@FD.registrar
def type_handler(
    name: str | None, fn: NonIterableTypeHandler
) -> NonIterableTypeHandler:
    name = _check_and_get_name_fn(name, fn)

    def load_into(lower: Ir2Verilog) -> None:
        def wrapper(*args, **kwargs):
            yield fn(*args, **kwargs)

        lower.register(name, wrapper)

    setattr(fn, "load_verilog", load_into)
    return fn


@FD.registrar
def type_handler_iterable(name: str | None, fn: TypeHandler) -> TypeHandler:
    name = _check_and_get_name_fn(name, fn)
    if name is None:
        name = fn.__name__

    def load_into(lower: Ir2Verilog) -> None:
        lower.register(name, fn)

    setattr(fn, "load_verilog", load_into)
    return fn
