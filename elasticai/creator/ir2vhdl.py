import re
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from typing import Any, Iterator, TypeAlias, TypeVar

from elasticai.creator.function_utils import KeyedFunctionDispatcher
from elasticai.creator.ir import Edge as _Edge
from elasticai.creator.ir import Graph, Lowerable, LoweringPass, RequiredField
from elasticai.creator.ir import Node as _Node
from elasticai.creator.ir.graph_iterators import bfs_iter_up
from elasticai.creator.ir.helpers import Shape, ShapeTuple
from elasticai.creator.lowering_pass_plugin import (
    IterableTypeHandlerDecorator,
    TypeHandlerDecorator,
)


class ShapeField(RequiredField[ShapeTuple, Shape]):
    def __init__(self):
        super().__init__(
            set_convert=lambda x: x.to_tuple(), get_convert=Shape.from_tuple
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
    input_shape: RequiredField[ShapeTuple, Shape] = ShapeField()
    output_shape: RequiredField[ShapeTuple, Shape] = ShapeField()


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
        dict(
            name=name,
            type=type,
            implementation=implementation,
            input_shape=to_tuple(input_shape),
            output_shape=to_tuple(output_shape),
        )
        | attributes
    )


class Edge(_Edge):
    src_sink_indices: tuple[tuple[int, int], ...]

    def __hash__(self) -> int:
        return hash((self.src, self.sink, self.src_sink_indices))


def edge(
    src: str, sink: str, src_sink_indices: Iterable[tuple[int, int]] | tuple[str, str]
) -> Edge:
    return Edge(dict(src=src, sink=sink, src_sink_indices=tuple(src_sink_indices)))


N = TypeVar("N", bound=VhdlNode)
E = TypeVar("E", bound=Edge)


class Implementation(Graph[N, E], Lowerable):
    def __init__(
        self,
        name: str,
        type: str,
        attributes: dict[str, Any],
        nodes=tuple(),
        edges=tuple(),
    ) -> None:
        super().__init__(nodes, edges)
        self._name = name
        self._type = type
        self.attributes = attributes

    @property
    def type(self) -> str:
        return self._type

    @property
    def name(self) -> str:
        return self._name

    def asdict(self) -> dict[str, Any]:
        return {
            "nodes": [n.data for n in self.nodes.values()],
            "edges": [e.data for e in self.edges.values()],
            "name": self.name,
            "type": self.type,
            "attributes": self.attributes,
        }

    @classmethod
    def fromdict(cls, data: dict[str, Any]) -> "Implementation":
        return cls(
            name=data["name"],
            type=data["type"],
            attributes=data["attributes"],
            nodes=map(VhdlNode, data["nodes"]),
            edges=map(Edge, data["edges"]),
        )

    def iterate_bfs_up_from(self, node: str) -> Iterator[N]:
        nodes = self.nodes
        for name in bfs_iter_up(self._g.get_predecessors, self._g.get_successors, node):
            yield nodes[name]


Code: TypeAlias = tuple[str, Sequence[str]]


class Ir2Vhdl(LoweringPass[Implementation, Code]):
    pass


type_handler: TypeHandlerDecorator[Implementation, Code] = TypeHandlerDecorator()

iterable_type_handler: IterableTypeHandlerDecorator[Implementation, Code] = (
    IterableTypeHandlerDecorator()
)


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
    def _search(cls, code: str) -> Signal | None:
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
    def _search(cls, code: str) -> Signal | None:
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
