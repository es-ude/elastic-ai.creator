import importlib.resources as res
import re
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from itertools import starmap
from typing import Iterator, TypeAlias

import elasticai.creator.function_dispatch as FD
import elasticai.creator.plugin as _pl
from elasticai.creator._hdl_ir import (
    Code,
    DataGraph,
    Node,
    NonIterableTypeHandler,
    Registry,
    Shape,
    TypeHandler,
    _check_and_get_name_fn,
)
from elasticai.creator._hdl_ir import (
    Edge as Edge,
)
from elasticai.creator._hdl_ir import (
    EdgeImpl as EdgeImpl,
)
from elasticai.creator._hdl_ir import (
    IrFactory as IrFactory,
)
from elasticai.creator._hdl_ir import (
    NodeImpl as NodeImpl,
)
from elasticai.creator._hdl_ir import (
    ShapeTuple as ShapeTuple,
)
from elasticai.creator.ir import ir_v2 as ir


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


class Signal(ABC):
    types: set[type["Signal"]] = set()

    @abstractmethod
    def define(self) -> Iterator[str]: ...

    @property
    @abstractmethod
    def name(self) -> str: ...

    @classmethod
    def from_code(cls, code: str) -> "Signal":
        for t in cls.types:
            if t.can_create_from_code(code):
                return t.from_code(code)
        return NullDefinedLogicSignal.from_code(code)

    @classmethod
    @abstractmethod
    def can_create_from_code(cls, code: str) -> bool: ...

    @classmethod
    def register_type(cls, t: type["Signal"]) -> None:
        cls.types.add(t)

    @abstractmethod
    def make_instance_specific(self, instance: str) -> "Signal": ...


class LogicSignal(Signal):
    def __init__(self, name: str):
        self._name = name

    def define(self) -> Iterator[str]:
        yield f"signal {self._name} : std_logic := '0';"

    @property
    def name(self) -> str:
        return self._name

    @classmethod
    def can_create_from_code(cls, code: str) -> bool:
        return cls._search(code) is not None

    @classmethod
    def _search(cls, code: str) -> re.Match[str] | None:
        match = re.search(
            r"signal ([a-zA-Z_][a-zA-Z0-9_]*)\s*:\s*std_logic(?:\s+|;)", code
        )
        return match

    @classmethod
    def from_code(cls, code: str) -> "Signal":
        match = cls._search(code)
        if match is None:
            raise ValueError(f"Cannot create signal from code: {code}")
        (name,) = match.groups()
        return cls(name)

    def make_instance_specific(self, instance: str) -> Signal:
        return self.__class__(f"{self.name}_{instance}")

    def __eq__(self, other: object) -> bool:
        if other is self:
            return True
        if isinstance(other, LogicSignal):
            return self._name == other._name
        return False


class LogicVectorSignal(Signal):
    def __init__(self, name: str, width: int):
        self._name = name
        self._width = width

    def define(self) -> Iterator[str]:
        yield f"signal {self._name} : std_logic_vector({self._width} - 1 downto 0) := (others => '0');"

    @property
    def name(self) -> str:
        return self._name

    @property
    def width(self) -> int:
        return self._width

    @classmethod
    def can_create_from_code(cls, code: str) -> bool:
        return cls._search(code) is not None

    @classmethod
    def _search(cls, code: str) -> re.Match[str] | None:
        match = re.match(
            r"signal ([a-zA-Z_][a-zA-Z0-9_]*)\s*: std_logic_vector\((\d+|(?:\d+ - \d+)) downto 0\)",
            code,
        )
        return match

    @classmethod
    def from_code(cls, code: str) -> "Signal":
        match = cls._search(code)
        if match is None:
            raise ValueError(f"Cannot create signal from code: {code}")
        name, width = match.groups()
        if " - " in width:
            a, b = width.split(" - ")
            width = str(int(a) - int(b))
        return cls(name, int(width) + 1)

    def make_instance_specific(self, instance: str) -> Signal:
        return self.__class__(f"{self.name}_{instance}", self.width)

    def __eq__(self, other) -> bool:
        if other is self:
            return True
        if isinstance(other, LogicVectorSignal):
            return self._name == other._name and self._width == other._width
        return False


class NullDefinedLogicSignal(Signal):
    def __init__(self, name):
        self._name = name

    def define(self) -> Iterator[str]:
        yield from []

    @property
    def name(self) -> str:
        return self._name

    @classmethod
    def can_create_from_code(cls, code: str) -> bool:
        return False

    @classmethod
    def from_code(cls, code: str) -> "Signal":
        return cls("<unknown>")

    def make_instance_specific(self, instance: str) -> Signal:
        return self


for t in (LogicSignal, LogicVectorSignal, NullDefinedLogicSignal):
    Signal.register_type(t)


class PortMap:
    def __init__(self, map: dict[str, Signal]):
        self._signals: dict[str, Signal] = map

    def as_dict(self) -> dict[str, str]:
        return {k: tuple(v.define())[0] for k, v in self._signals.items()}

    @classmethod
    def from_dict(cls, data: dict[str, str]) -> "PortMap":
        return cls({k: Signal.from_code(v) for k, v in data.items()})

    def __eq__(self, other: object) -> bool:
        if other is self:
            return True
        if isinstance(other, PortMap):
            return self._signals == other._signals
        return False


class Instance:
    """Represents an entity that we can/want instantiate.

    The aggregates all the knowledge that is necessary to
    instantiate and use the corresponding entity programmatically,
    when generating vhdl code.
    """

    def __init__(
        self,
        node: Node,
        generic_map: dict[str, str],
        port_map: dict[str, Signal],
    ):
        self._node = node
        self._generics: dict[str, str] = {k.lower(): v for k, v in generic_map.items()}
        self.port_map = {
            k: v.make_instance_specific(self._node.name) for k, v in port_map.items()
        }

    @property
    def input_shape(self) -> Shape:
        return self._node.input_shape

    @property
    def output_shape(self) -> Shape:
        return self._node.output_shape

    @property
    def name(self) -> str:
        return self._node.name

    @property
    def implementation(self) -> str:
        return self._node.implementation

    def define_signals(self) -> Iterator[str]:
        for s in self.port_map.values():
            yield from s.define()

    def instantiate(self) -> Iterator[str]:
        yield from (f"{self.name}: entity work.{self.implementation}(rtl) ",)
        generics = tuple(self._generics.items())
        if len(generics) > 0:
            yield "generic map ("
            for key, value in generics[:-1]:
                yield f"  {key.upper()} => {value},"
            for g in generics[-1:]:
                yield f"  {g[0].upper()} => {g[1]}"
            yield "  )"
        port_map = tuple(self.port_map.items())
        yield "  port map ("

        for k, v in port_map[:-1]:
            yield f"    {k} => {v.name},"
        for k, v in port_map[-1:]:
            yield f"    {k} => {v.name}"

        yield "  );"


def _check_and_get_fn_name(name: str | None, fn: Callable) -> str:
    if name is None:
        if hasattr(fn, "__name__") and isinstance(fn.__name__, str):
            name = fn.__name__
        else:
            raise Exception(
                "provided type handler has to be a function or named explicitly"
            )
    return name


class InstanceFactory:
    """Automatically creates Instances from VhdlNodes based on their `type` field."""

    @FD.dispatch_method(str)
    def _handle_type(self, fn: Callable[[Node], Instance], node: Node) -> Instance:
        return fn(node)

    @_handle_type.key_from_args
    def _get_type_from_node(self, node: Node) -> str:
        return node.type

    @FD.registrar_method
    def register(
        self,
        type: str | None,
        fn: Callable[[Node], Instance],
    ) -> Callable[[Node], Instance]:
        type = _check_and_get_name_fn(type, fn)
        return self._handle_type.register(type, fn)

    def __call__(self, node: Node) -> Instance:
        return self._handle_type(node)


PluginSymbol: TypeAlias = _pl.PluginSymbolFn[Ir2Vhdl, [DataGraph, Registry], Code]
PluginSymbolIter: TypeAlias = _pl.PluginSymbolFn[
    Ir2Vhdl, [DataGraph, Registry], Iterable[Code]
]


class PluginLoader(_pl.PluginLoader[Ir2Vhdl]):
    """Plugin loader for ir2vhdl translation."""

    def __init__(self, lowering: Ir2Vhdl):
        builder: _pl.SymbolFetcherBuilder[PluginSpec, Ir2Vhdl] = (
            _pl.SymbolFetcherBuilder(PluginSpec)
        )
        fetcher: _pl.SymbolFetcher[Ir2Vhdl] = (
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
    def __get_generated(plugin: PluginSpec) -> Iterator[_pl.PluginSymbol]:
        if plugin.target_runtime == "vhdl":
            yield from _pl.import_symbols(plugin.package, plugin.generated)


class _StaticFile(_pl.PluginSymbol[Ir2Vhdl]):
    def __init__(self, name: str, package: str):
        self._name = name
        self._package = package

    @property
    def name(self) -> str:
        return self._name

    def load_into(self, receiver: Ir2Vhdl):
        receiver.register_static(self.name, self)

    @classmethod
    def make_symbols(cls, p: PluginSpec) -> Iterator[_pl.PluginSymbol[Ir2Vhdl]]:
        if p.target_runtime == "vhdl":
            for name in p.static_files:
                yield cls(name=name, package=p.package)

    def __call__(self) -> str:
        file = res.files(self._package).joinpath(f"vhdl/{self.name}")
        return file.read_text()


@FD.registrar
def type_handler(name: str | None, fn: NonIterableTypeHandler) -> PluginSymbol:
    name = _check_and_get_name_fn(name, fn)

    def load_into(lower: Ir2Vhdl) -> None:
        def wrapper(*args, **kwargs):
            yield fn(*args, **kwargs)

        lower.register(name, wrapper)

    return _pl.make_plugin_symbol(load_into, fn)


@FD.registrar
def type_handler_iterable(name: str | None, fn: TypeHandler) -> PluginSymbolIter:
    name = _check_and_get_name_fn(name, fn)
    if name is None:
        name = fn.__name__

    def load_into(lower: Ir2Vhdl) -> None:
        lower.register(name, fn)

    return _pl.make_plugin_symbol(load_into, fn)
