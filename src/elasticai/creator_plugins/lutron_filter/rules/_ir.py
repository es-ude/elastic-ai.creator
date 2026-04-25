import re
from abc import abstractmethod
from collections.abc import Callable
from itertools import chain
from typing import Concatenate, Iterable, Protocol, cast

import elasticai.creator.ir as ir
from elasticai.creator.graph import find_all_subgraphs
from elasticai.creator.ir import (
    AttributeMapping,
    Edge,
    IrFactory,
    IrSerializer,
    PatternRule,
    PatternRuleSpec,
    Rule,
    attribute,
)
from elasticai.creator.ir import DataGraph as _BaseDataGraph
from elasticai.creator.ir import (
    DataGraphImpl as _DGraphImpl,
)
from elasticai.creator.ir import (
    EdgeImpl as _EdgeImpl,
)
from elasticai.creator.ir import GraphImpl as _GraphImpl
from elasticai.creator.ir import Node as _Node
from elasticai.creator.ir import NodeEdgeFactory as _NodeEdgeFactory
from elasticai.creator.ir import (
    NodeImpl as _NodeImpl,
)
from elasticai.creator.ir import Registry as Registry
from elasticai.creator_plugins.grouped_filter import FilterParameters

_ir_serializer: IrSerializer = IrSerializer()


class NameRegistry:
    def __init__(self) -> None:
        self._registry = {}

    def _get_name_count(self, name) -> int:
        return self._registry.get(name, 0)

    def prepopulate(self, names) -> "NameRegistry":
        for name in names:
            match = re.match(r"(.+)_(\d+)$", name)
            suffix = 0
            if match:
                name = match.group(1)
                suffix = int(match.group(2))
            suffix = max(suffix, self._get_name_count(name))
            self._registry[name] = suffix
        return self

    def get_unique_name(self, name) -> str:
        if name not in self._registry:
            self._registry[name] = 0
            return name

        new_name = f"{name}_{self._registry[name] + 1}"
        self._registry[name] += 1
        return new_name


class Node(_NodeImpl, _Node):
    @property
    def implementation(self) -> str:
        item = self._attributes.get("implementation", "<none>")
        if not isinstance(item, str):
            raise TypeError(
                f"expected implementation to have to type `str` but found {type(item)}"
            )
        return item


class _DataGraph[N: _Node, E: Edge](_BaseDataGraph[N, E], Protocol):
    @property
    @abstractmethod
    def type(self) -> str: ...


type DataGraph = _DataGraph[Node, Edge]


class _DataGraphImpl[N: _Node, E: Edge](_DGraphImpl[N, E]):
    @property
    def type(self) -> str:
        result = self.attributes.get("type", "<undefined>")
        if not isinstance(result, str):
            raise TypeError()
        return result


class Decoratable(Protocol):
    @property
    def attributes(self) -> AttributeMapping: ...
    @property
    def type(self) -> str: ...


class FilterDecorator[T: Decoratable]:
    def __init__(self, decorated: T) -> None:
        self._decorated = decorated

    @property
    def decorated(self) -> T:
        return self._decorated

    def call_on_decorated[**P, R](
        self, fn: Callable[Concatenate[T, P], T], *args: P.args, **kwargs: P.kwargs
    ) -> "FilterDecorator[T]":
        return FilterDecorator(fn(self._decorated, *args, **kwargs))

    @property
    def attributes(self) -> AttributeMapping:
        return self._decorated.attributes

    def _unify_kernel_size(self, kernel_size: int | tuple[int] | list[int]) -> int:
        if not isinstance(kernel_size, int):
            if len(kernel_size) != 1:
                raise ValueError(f"kernel size {kernel_size} not supported.")
            return kernel_size[0]
        return kernel_size

    def _get_conv_filter_params(self) -> FilterParameters:
        return FilterParameters(
            kernel_size=self._unify_kernel_size(self.attributes["kernel_size"]),
            out_channels=self.attributes["out_channels"],
            in_channels=self.attributes["in_channels"],
            groups=cast(int, self.attributes.get("groups", 1)),
            stride=cast(int, self.attributes.get("stride", 1)),
        )

    @property
    def type(self) -> str:
        return self.decorated.type

    @property
    def kernel_size(self) -> int:
        return self.filter_parameters.kernel_size

    @property
    def groups(self) -> int:
        return self.filter_parameters.groups

    @property
    def in_channels(self) -> int:
        return self.filter_parameters.in_channels

    @property
    def out_channels(self) -> int:
        return self.filter_parameters.out_channels

    @property
    def filter_parameters(self) -> FilterParameters:
        if not hasattr(self, "__cached_filter_params"):
            match self.type:
                case "conv1d":
                    p = self._get_conv_filter_params()
                case _:
                    raise TypeError(f"unsupported type '{self.type}' for FilterNode")
            self.__cached_filter_params = p
        return self.__cached_filter_params


class NodeEdgeFactory(_NodeEdgeFactory[Node, Edge]):
    def edge(
        self, src: str, dst: str, attributes: AttributeMapping = AttributeMapping(), /
    ) -> Edge:
        return _EdgeImpl(src, dst, attributes)

    def node(
        self, name: str, attributes: AttributeMapping = AttributeMapping(), /
    ) -> Node:
        return Node(name, attributes)


def wrap_graph(g: _BaseDataGraph[_Node, Edge]) -> _DataGraph[Node, Edge]:
    return _DataGraphImpl(
        factory=NodeEdgeFactory(),
        attributes=g.attributes,
        graph=g.graph,
        node_attributes=g.node_attributes,
    )


def wrap_node(n: _Node) -> Node:
    return Node(n.name, n.attributes)


def wrap_registry(reg: Registry) -> Registry[DataGraph]:
    return ir.Registry((name, wrap_graph(g)) for name, g in reg.items())


class _IrFactory(NodeEdgeFactory):
    def graph(self, attributes: AttributeMapping = AttributeMapping(), /) -> DataGraph:
        return _DataGraphImpl(
            factory=self,
            attributes=attributes,
            node_attributes=AttributeMapping(),
            graph=_GraphImpl(lambda: AttributeMapping()),
        )


ir_factory: IrFactory[Node, Edge, DataGraph] = _IrFactory()


def serialize(g: DataGraph) -> dict:
    return _ir_serializer.serialize(g)


type NodeConstraint = Callable[[Node, Node], bool]


def make_default_constraint(_: Registry[DataGraph], /) -> NodeConstraint:
    def default_constraint(pattern_node: Node, graph_node: Node, /) -> bool:
        if pattern_node.type == "interface":
            return True
        else:
            return pattern_node.type == graph_node.type

    return default_constraint


def node(name: str, type: str, implementation: str | None = None) -> Node:
    if implementation is None:
        return ir_factory.node(name, attribute(type=type))
    return ir_factory.node(name, attribute(type=type, implementation=implementation))


def sequential(*sequence: tuple[str, str] | str) -> DataGraph:
    def _to_tuple(arg: str | tuple[str, str]) -> tuple[str, str]:
        if isinstance(arg, str):
            return arg, arg
        else:
            return arg

    _sequence = tuple(map(_to_tuple, sequence))
    nodes = tuple((node(*args) for args in _sequence))
    node_seq = tuple(name for name, _ in _sequence)
    edges = zip(node_seq[:-1], node_seq[1:])
    return ir_factory.graph().add_nodes(*nodes).add_edges(*edges)


def sequential_with_interface(*sequence: tuple[str, str] | str) -> DataGraph:
    sequence_parts: list[Iterable[tuple[str, str] | str]] = [
        [("start", "interface")],
        sequence,
        [("end", "interface")],
    ]
    seq_with_interface = tuple(chain.from_iterable(sequence_parts))
    return sequential(*seq_with_interface)


class Pattern(ir.Pattern):
    def __init__(
        self,
        graph: ir.DataGraph,
        constraint_factory: Callable[[Registry[DataGraph]], NodeConstraint],
    ):
        self._g = graph
        self._constraint_factory = constraint_factory

    @property
    def graph(self) -> ir.DataGraph:
        return self._g

    @property
    def interface(self) -> set[str]:
        return {"start", "end"}

    def match(self, g: ir.DataGraph, registry: ir.Registry, /) -> list[dict[str, str]]:
        wg = wrap_graph(g)
        wp = wrap_graph(self.graph)
        _constraint_fn = self._constraint_factory(wrap_registry(registry))

        def constraint(pattern_node: str, graph_node: str) -> bool:
            return _constraint_fn(
                wp.nodes[pattern_node],
                wg.nodes[graph_node],
            )

        return find_all_subgraphs(self.graph, g, constraint)


def pattern_rule(
    graph: _BaseDataGraph[_Node, Edge],
    replacement_fn: Callable[
        [DataGraph, Registry[DataGraph]], tuple[DataGraph, Registry[DataGraph]]
    ],
    make_node_constraint: Callable[
        [ir.Registry[DataGraph]], NodeConstraint
    ] = make_default_constraint,
    interface=("start", "end"),
) -> Rule:
    def wrap_replace(
        match: ir.DataGraph, registry: ir.Registry
    ) -> tuple[ir.DataGraph, ir.Registry]:
        return replacement_fn(wrap_graph(match), wrap_registry(registry))

    return PatternRule(
        spec=PatternRuleSpec(
            pattern=Pattern(graph, make_default_constraint),
            replacement_fn=wrap_replace,
        )
    )


def build_sequential_ir(
    registry: dict[str, AttributeMapping], sequence: tuple[str, ...]
) -> tuple[DataGraph, Registry[DataGraph]]:
    node_names = list(map(str, range(len(sequence))))
    nodes = []
    _registry: Registry[DataGraph] = Registry(
        **{k: ir_factory.graph(v) for k, v in registry.items()}
    )
    for name, implementation in zip(node_names, sequence):
        nodes.append(node(name, _registry[implementation].type, implementation))
    nodes = [node("input", "input", "<none>")] + nodes
    nodes.append(node("output", "output", "<none>"))
    root = (
        ir_factory.graph()
        .add_nodes(*nodes)
        .add_edges(*((src.name, dst.name) for src, dst in zip(nodes[:-1], nodes[1:])))
    )
    return root, _registry
