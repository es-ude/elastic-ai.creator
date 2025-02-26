from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from importlib import resources as res
from typing import Protocol, TypeAlias

import elasticai.creator.function_utils as F
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


def _type_handler(
    name: str, fn: TypeHandlerFn
) -> pl.PluginSymbolFn[Ir2Verilog, [Implementation], Code]:
    def load_into(lower: Ir2Verilog) -> None:
        lower.register(name)(fn)

    return pl.make_plugin_symbol(load_into, fn)


def _type_handler_for_iterable(
    name: str, fn: Callable[[Implementation], Iterable[Code]]
) -> pl.PluginSymbolFn[Ir2Verilog, [Implementation], Iterable[Code]]:
    def load_into(lower: Ir2Verilog) -> None:
        lower.register_iterable(name)(fn)

    return pl.make_plugin_symbol(load_into, fn)


type_handler = F.FunctionDecorator(_type_handler)
type_handler_iterable = F.FunctionDecorator(_type_handler_for_iterable)
