import warnings
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from itertools import starmap
from typing import Any, Protocol, override

import elasticai.creator.function_dispatch as FD
import elasticai.creator.ir as ir
import elasticai.creator.plugin as _pl
from elasticai.creator.hdl_ir import (
    Code,
    DataGraph,
    NonIterableTypeHandler,
    Registry,
    TypeHandler,
    _check_and_get_name_fn,
)
from elasticai.creator.hdl_ir import (
    Edge as Edge,
)
from elasticai.creator.hdl_ir import (
    EdgeImpl as EdgeImpl,
)
from elasticai.creator.hdl_ir import (
    IrFactory as IrFactory,
)
from elasticai.creator.hdl_ir import (
    NodeImpl as NodeImpl,
)
from elasticai.creator.hdl_ir import (
    ShapeTuple as ShapeTuple,
)
from elasticai.creator.plugin import PluginLoaderBase, StaticFileBase


@dataclass
class PluginSpec(_pl.PluginSpec):
    generated: tuple[str, ...]
    static_files: tuple[str, ...]


class Ir2Vhdl:
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
                yield f"{name}.vhd", code
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


class PluginSymbolObject(Protocol):
    def load_vhdl(self, receiver: Ir2Vhdl) -> None: ...


class PluginSymbolClass(Protocol):
    @classmethod
    def load_vhdl(cls, receiver: Ir2Vhdl) -> None: ...


class PluginSymbolStatic(Protocol):
    @staticmethod
    def load_vhdl(receiver: Ir2Vhdl) -> None: ...


type PluginSymbol = PluginSymbolClass | PluginSymbolObject | PluginSymbolStatic


class _StaticFileSymbol:
    def __init__(self, file: StaticFileBase):
        self._file = file

    def load_vhdl(self, receiver: Ir2Vhdl) -> None:
        receiver.register_static(self._file.name, self._file.get_content)


class PluginLoader(PluginLoaderBase):
    """PluginLoader for Ir2Vhdl passes."""

    def __init__(self, lowering: Ir2Vhdl):
        self._receiver = lowering
        super().__init__(PluginSpec)

    @override
    def filter_plugin_dicts(
        self, plugins: Iterable[dict[str, Any]]
    ) -> Iterable[dict[str, Any]]:
        for p in plugins:
            if p["target_runtime"] == "vhdl":
                yield p

    @override
    def get_symbols(self, specs: Iterable[PluginSpec]) -> Iterable[PluginSymbol]:
        for spec in specs:
            yield from _pl.import_symbols(spec.package, spec.generated)
            for static_name in spec.static_files:
                yield _StaticFileSymbol(
                    StaticFileBase(static_name, spec.package, "vhdl")
                )

    @override
    def load_symbol(self, symbol: PluginSymbol) -> None:
        if hasattr(symbol, "load_vhdl"):
            symbol.load_vhdl(self._receiver)
        elif hasattr(symbol, "load_into"):
            warnings.warn(
                "Loading legacy plugin symbol, this behaviour will be removed in the future ensure your plugin symbol provides a load_vhdl method",
                stacklevel=2,
                category=DeprecationWarning,
            )
            symbol.load_into(self._receiver)
        else:
            raise TypeError("Failed to load plugin symbol")


@FD.registrar
def type_handler(
    name: str | None, fn: NonIterableTypeHandler
) -> NonIterableTypeHandler:
    name = _check_and_get_name_fn(name, fn)

    def load_into(lower: Ir2Vhdl) -> None:
        def wrapper(*args, **kwargs):
            yield fn(*args, **kwargs)

        lower.register(name, wrapper)

    setattr(fn, "load_vhdl", load_into)
    return fn


@FD.registrar
def type_handler_iterable(name: str | None, fn: TypeHandler) -> TypeHandler:
    name = _check_and_get_name_fn(name, fn)
    if name is None:
        name = fn.__name__

    def load_into(lower: Ir2Vhdl) -> None:
        lower.register(name, fn)

    setattr(fn, "load_vhdl", load_into)
    return fn
