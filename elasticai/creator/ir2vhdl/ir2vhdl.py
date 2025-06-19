import importlib.resources as res
import operator
import re
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from functools import reduce
from typing import Any, Iterator, TypeAlias, TypeGuard, TypeVar, overload

import elasticai.creator.function_utils as F
import elasticai.creator.plugin as _pl
from elasticai.creator.function_utils import KeyedFunctionDispatcher
from elasticai.creator.graph import BaseGraph
from elasticai.creator.ir import (
    Attribute,
    LoweringPass,
    RequiredField,
)
from elasticai.creator.ir import Edge as _Edge
from elasticai.creator.ir import (
    Implementation as _Implementation,
)
from elasticai.creator.ir import Node as _Node
from elasticai.creator.plugin import PluginLoader as _Loader
from elasticai.creator.plugin import PluginSpec as _PluginSpec
from elasticai.creator.plugin import PluginSymbol as _PluginSymbol


@dataclass
class PluginSpec(_PluginSpec):
    generated: tuple[str, ...]
    static_files: tuple[str, ...]


ShapeTuple: TypeAlias = tuple[int] | tuple[int, int] | tuple[int, int, int]


def is_shape_tuple(values) -> TypeGuard[ShapeTuple]:
    max_num_values = 3
    return len(values) <= max_num_values


class Shape:
    @overload
    def __init__(self, width: int, /) -> None: ...

    @overload
    def __init__(self, depth: int, width: int, /) -> None: ...

    @overload
    def __init__(self, depth: int, width: int, height: int, /) -> None: ...

    def __init__(self, *values: int) -> None:
        """values are interpreted as one of the following:
        - width
        - depth, width
        - depth, width, height

        Usually width is kernel_size, depth is channels
        """

        if is_shape_tuple(values):
            self._data = values
        else:
            raise TypeError(f"taking at most three ints, given {values}")

    @classmethod
    def from_tuple(cls, values: ShapeTuple | list[int]) -> "Shape":
        return cls(*values)  # type ignore

    def to_tuple(self) -> ShapeTuple:
        return self._data

    def to_list(self) -> list[int]:
        return list(self.to_tuple())

    def __getitem__(self, item):
        return self._data[item]

    def size(self) -> int:
        return reduce(operator.mul, self._data, 1)

    def ndim(self) -> int:
        return len(self._data)

    @property
    def depth(self) -> int:
        if len(self._data) > 1:
            return self._data[0]
        return 1

    def __eq__(self, other):
        if isinstance(other, tuple):
            return self._data == other
        if isinstance(other, Shape):
            return self._data == other._data

        return False

    @property
    def width(self) -> int:
        if len(self._data) > 1:
            return self._data[1]
        else:
            return self._data[0]

    @property
    def height(self) -> int:
        if len(self._data) > 2:
            return self._data[2]
        return 1

    def __repr__(self) -> str:
        match self._data:
            case (width,):
                return f"Shape({width=})"
            case (depth, width):
                return f"Shape({depth=}, {width=})"
            case (depth, width, height):
                return f"Shape({depth=}, {width=}, {height=})"
            case _:
                return f"Shape({self._data})"


class ShapeField(RequiredField[list[int], Shape]):
    def __init__(self):
        super().__init__(
            set_convert=lambda x: x.to_list(), get_convert=Shape.from_tuple
        )


class VhdlNode(_Node):
    """Extending ir.core.Node to a vhdl specific node.

    `VhdlNode` contains all knowledge that we need to create
    and use an instance of a vhdl entity. However, this becomes
    a little bit complicated because vhdl differentiates between
    the *entity* and the *architecture* of a component.
    The entity is similar to an _interface_ while the architecture
    is similar to the _implementation_. However, to instantiate
    components, we need to know both names.

    Attributes:

    implementation:: The name of the implementation will be used to derive the architecture name.
        E.g., if the implementation is `"adder"`, we will instantiate the entity `work.adder(rtl)`.
        CAUTION: This behaviour is subject to change. Future versions might require the full entity name
    """

    implementation: str
    input_shape: RequiredField[list[int], Shape] = ShapeField()
    output_shape: RequiredField[list[int], Shape] = ShapeField()


def vhdl_node(
    name: str,
    type: str,
    implementation: str,
    input_shape: Shape | ShapeTuple,
    output_shape: Shape | ShapeTuple,
    attributes: dict | None = None,
) -> VhdlNode:
    """Convenience method for creating new vhdl nodes."""
    if attributes is None:
        attributes = {}

    def to_tuple(s: Shape | ShapeTuple) -> ShapeTuple:
        if isinstance(s, Shape):
            return s.to_tuple()
        else:
            return s

    return VhdlNode(
        name=name,
        data=dict(
            type=type,
            implementation=implementation,
            input_shape=to_tuple(input_shape),
            output_shape=to_tuple(output_shape),
        )
        | attributes,
    )


class Edge(_Edge):
    src_dst_indices: tuple[tuple[int, int] | tuple[str, str], ...]

    def __hash__(self) -> int:
        return hash((self.src, self.dst, self.src_dst_indices))


def edge(
    src: str, dst: str, src_dst_indices: Iterable[tuple[int, int]] | tuple[str, str]
) -> Edge:
    return Edge(src=src, dst=dst, data={"src_dst_indices": tuple(src_dst_indices)})


N = TypeVar("N", bound=VhdlNode)
E = TypeVar("E", bound=Edge)


class Implementation(_Implementation[N, E]):
    name: str
    type: str

    @overload
    def __init__(
        self: "Implementation[VhdlNode, Edge]",
        *,
        name: str | None = None,
        type: str | None = None,
        data: dict[str, Attribute] | None = None,
        attributes: dict[str, Attribute] | None = None,
    ) -> None: ...

    @overload
    def __init__(
        self: "Implementation[N, Edge]",
        *,
        node_fn: Callable[[dict], N],
        name: str | None = None,
        type: str | None = None,
        data: dict[str, Attribute] | None = None,
        attributes: dict[str, Attribute] | None = None,
        graph: BaseGraph | None = None,
    ) -> None: ...

    @overload
    def __init__(
        self: "Implementation[VhdlNode, E]",
        *,
        edge_fn: Callable[[dict], E],
        name: str | None = None,
        type: str | None = None,
        data: dict[str, Attribute] | None = None,
        attributes: dict[str, Attribute] | None = None,
        graph: BaseGraph | None = None,
    ) -> None: ...

    @overload
    def __init__(
        self: "Implementation[N, E]",
        *,
        node_fn: Callable[[dict], N],
        edge_fn: Callable[[dict], E],
        name: str | None = None,
        type: str | None = None,
        data: dict[str, Attribute] | None = None,
        attributes: dict[str, Attribute] | None = None,
        graph: BaseGraph | None = None,
    ) -> None: ...

    def __init__(
        self,
        *,
        name: str | None = None,
        type: str | None = None,
        node_fn=VhdlNode,
        edge_fn=Edge,
        data: dict[str, Any] | None = None,
        attributes: dict[str, Any] | None = None,
        graph: BaseGraph | None = None,
    ) -> None:
        if attributes is not None and data is not None:
            raise TypeError("pass either attributes or data argument")
        if attributes is not None:
            warnings.warn(
                "the argument `attributes` is deprecated, use `data` instead.",
                DeprecationWarning,
                2,
            )
            data = attributes

        if graph is None:
            graph = BaseGraph()
        super().__init__(
            node_fn=node_fn,
            edge_fn=edge_fn,
            data=data,
            graph=graph,
        )
        if name is not None:
            self.data["name"] = name
        if type is not None:
            self.data["type"] = type


Code: TypeAlias = tuple[str, Sequence[str]]


class Ir2Vhdl(LoweringPass[Implementation, Code]):
    def __init__(self) -> None:
        super().__init__()
        self.__static_files: dict[str, Callable[[], str]] = {}

    def register_static(self, name: str, fn: Callable[[], str]) -> None:
        self.__static_files[name] = fn

    def __call__(self, args: Iterable[Implementation]) -> Iterator[Code]:
        for name, content in super().__call__(args):
            yield f"{name}.vhd", content
        for name, fn in self.__static_files.items():
            yield name, [fn()]


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
        node: VhdlNode,
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


class InstanceFactory(KeyedFunctionDispatcher[VhdlNode, Instance]):
    """Automatically creates Instances from VhdlNodes based on their `type` field."""

    def __init__(self):
        def dispatch_key_fn(node: VhdlNode) -> str:
            return node.type

        super().__init__(dispatch_key_fn=dispatch_key_fn)


PluginSymbol: TypeAlias = _PluginSymbol[Ir2Vhdl]


class PluginLoader(_Loader[Ir2Vhdl]):
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
    def __get_generated(plugin: PluginSpec) -> Iterator[PluginSymbol]:
        if plugin.target_runtime == "vhdl":
            yield from _pl.import_symbols(plugin.package, plugin.generated)


class _StaticFile(_PluginSymbol[Ir2Vhdl]):
    def __init__(self, name: str, package: str):
        self._name = name
        self._package = package

    @property
    def name(self) -> str:
        return self._name

    def load_into(self, receiver: Ir2Vhdl):
        receiver.register_static(self.name, self)

    @classmethod
    def make_symbols(cls, p: PluginSpec) -> Iterator[PluginSymbol]:
        if p.target_runtime == "vhdl":
            for name in p.static_files:
                yield cls(name=name, package=p.package)

    def __call__(self) -> str:
        file = res.files(self._package).joinpath(f"vhdl/{self.name}")
        return file.read_text()


_Tcontra = TypeVar("_Tcontra", contravariant=True)

TypeHandlerFn: TypeAlias = Callable[[Implementation], Code]


def _type_handler(name: str, fn: TypeHandlerFn) -> PluginSymbol:
    def load_into(lower: Ir2Vhdl) -> None:
        lower.register(name)(fn)

    return _pl.make_plugin_symbol(load_into, fn)


def _type_handler_for_iterable(
    name: str, fn: Callable[[Implementation], Iterable[Code]]
) -> PluginSymbol:
    def load_into(lower: Ir2Vhdl) -> None:
        lower.register_iterable(name)(fn)

    return _pl.make_plugin_symbol(load_into, fn)


type_handler = F.FunctionDecorator(_type_handler)
type_handler_iterable = F.FunctionDecorator(_type_handler_for_iterable)
