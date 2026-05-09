from collections.abc import Callable, Iterable, Iterator, Mapping
from typing import Protocol, cast

import elasticai.creator.function_dispatch as FD
from elasticai.creator import ir
from elasticai.creator.graph import bfs_iter_down

type DataGraph = ir.DataGraph[ir.Node, ir.Edge]
type Registry = ir.Registry[DataGraph]
type TypeHandler = Callable[[DataGraph, tuple[tuple[int, ...], ...]], tuple[int, ...]]
type Shape = tuple[int, ...]


def _guard_shape(item: object) -> Shape:
    return _type_check(item, tuple)


class ShapedEdge(ir.Edge, Protocol):
    @property
    def shape(self) -> Shape: ...


class Node(ir.Node, Protocol):
    @property
    def implementation(self) -> str: ...


type ShapedDatagraph = ir.DataGraph[Node, ShapedEdge]


class IrShapeInference:
    def __init__(self) -> None:
        self._ir_factory = IrFactory()
        self._registry: Mapping[str, DataGraph] = ir.Registry()
        self._root = self._ir_factory.graph(ir.attribute(type="module"))
        self._original = ir.DefaultIrFactory().graph()
        self._unfinished_nodes: set[str] = set()

    def __call__(
        self,
        root: DataGraph,
        reg: Mapping[str, DataGraph],
        input_node_shapes: dict[str, tuple[int, ...]],
    ) -> ShapedDatagraph:
        self._original = root
        self._registry = reg
        self._root = self._ir_factory.graph(ir.attribute(type="module"))
        self._check_input_shape_nodes_are_valid(input_node_shapes.keys())
        self._initialize_input_nodes(input_node_shapes)
        self._walk_graph_and_compute_other_shapes()
        self._root = self._root.add_nodes(*(self._original.nodes.values()))
        return self._root

    def _walk_graph_and_compute_other_shapes(self) -> None:
        for node in self._get_non_input_nodes_in_data_dependency_order():
            edges = list(self._collect_incoming_edges(node.name))
            input_shapes = tuple(e.shape for e in edges)
            output_shape = self._get_out_shape_for_node(node, input_shapes)
            self._add_outgoing_edges(node.name, output_shape)

    def _should_look_up_in_shapes_from_nodes_handlers(self, node: ir.Node):
        return (
            "implementation" not in node.attributes
            or node.type == "function"
            or self._get_node_impl(node).attributes["type"]
            not in self._get_out_shape_from_dgraph.registry
        )

    def _get_out_shape_for_node(
        self, node: Node, input_shapes: tuple[Shape, ...]
    ) -> Shape:
        if self._should_look_up_in_shapes_from_nodes_handlers(node):
            return self._get_out_shape_from_node(node, input_shapes)

        impl = self._get_node_impl(node.name)
        return self._get_out_shape_from_dgraph(impl, input_shapes)

    def _add_outgoing_edges(self, node: str, shape: Shape) -> None:
        edges = []
        for succ, attrs in self._original.successors[node].items():
            edges.append((node, succ, attrs | ir.attribute(shape=shape)))
        self._root = self._root.add_edges(*edges)

    def _collect_incoming_edges(self, node: str) -> Iterator[ShapedEdge]:
        for pred in self._root.predecessors[node]:
            yield self._root.edges[(pred, node)]

    def _get_non_input_nodes_in_data_dependency_order(self) -> Iterator[Node]:
        input_nodes: set[str] = set()
        for node in self._original.nodes.values():
            if node.type == "input":
                input_nodes.add(node.name)
        names = list(
            bfs_iter_down(
                successors=self._original.successors.__getitem__,
                predecessors=self._original.predecessors.__getitem__,
                start=input_nodes,
            )
        )
        names = list(names)
        for name in names:
            node = self._original.nodes[name]
            if node.type != "output":
                yield self._ir_factory.node(node.name, node.attributes)

    def _check_input_shape_nodes_are_valid(self, input_nodes: Iterable[str]) -> None:
        for input_node in input_nodes:
            node = self._original.nodes[input_node]
            if node.type != "input":
                raise Exception(f"{input_node} is not a input")

    def _initialize_input_nodes(self, input_node_shapes: dict[str, Shape]) -> None:
        edges_from_input_nodes = []
        for n, shape in input_node_shapes.items():
            for succ, attrs in self._original.successors[n].items():
                edges_from_input_nodes.append(
                    (n, succ, attrs | ir.attribute(shape=shape))
                )
        for e in edges_from_input_nodes:
            self._root = self._root.add_edge(*e)

    def _get_node_impl(self, node: ir.Node | str) -> DataGraph:
        if isinstance(node, str):
            node = self._original.nodes[node]

        return self._registry[node.attributes["implementation"]]

    @FD.dispatch_method(str)
    def _get_out_shape_from_dgraph(
        self,
        fn: TypeHandler,
        graph: DataGraph,
        input_shapes: tuple[Shape, ...],
    ) -> Shape:
        return fn(graph, input_shapes)

    @_get_out_shape_from_dgraph.key_from_args
    def _get_type_from_dg(
        self, graph: DataGraph, input_shapes: tuple[Shape, ...]
    ) -> str:
        return cast(str, graph.attributes["type"])

    @FD.dispatch_method(str)
    def _get_out_shape_from_node(
        self,
        fn: Callable[[Node, tuple[Shape, ...]], Shape],
        node: Node,
        input_shapes: tuple[Shape, ...],
    ) -> Shape:
        return fn(node, input_shapes)

    @_get_out_shape_from_node.key_from_args
    def _get_node_type(self, node: ir.Node, input_shapes: tuple[Shape, ...]) -> str:
        if node.type == "function":
            return node.attributes["implementation"]
        return node.type

    @FD.registrar_method
    def register_dgraph(self, key: str | None, fn: TypeHandler) -> TypeHandler:
        key = self._check_and_get_name(key, fn)
        self._get_out_shape_from_dgraph.register(key, fn)
        return fn

    @FD.registrar_method
    def register_node(
        self, key: str | None, fn: Callable[[Node, tuple[Shape, ...]], Shape]
    ) -> Callable[[Node, tuple[Shape, ...]], Shape]:
        key = self._check_and_get_name(key, fn)
        self._get_out_shape_from_node.register(key, fn)
        return fn

    @staticmethod
    def _check_and_get_name(name: str | None, fn: Callable) -> str:
        if name is None:
            if hasattr(fn, "__name__") and isinstance(fn.__name__, str):
                return fn.__name__
            else:
                raise TypeError(
                    "specify the type handler's type explicitly if you want to register a non-function callable"
                )
        return name


def _type_check[T](item: object, t: type[T]) -> T:
    if isinstance(item, t):
        return item
    else:
        raise TypeError(f"Expected type {t} but found {type(item)}")


class _NodeImpl(ir.NodeImpl):
    @property
    def implementation(self) -> str:
        if "implementation" not in self.attributes:
            return "<undefined>"
        return _type_check(self.attributes["implementation"], str)


class _ShapedEdgeImpl(ir.EdgeImpl):
    @property
    def shape(self) -> Shape:
        if "shape" not in self.attributes:
            raise ValueError(f"node {self.src} -> {self.dst} has no shape")
        return _guard_shape(self.attributes["shape"])


class IrFactory(ir.StdIrFactory[Node, ShapedEdge, ShapedDatagraph]):
    def __init__(self) -> None:
        super().__init__(
            node_fn=_NodeImpl, edge_fn=_ShapedEdgeImpl, graph_fn=ir.DataGraphImpl
        )
