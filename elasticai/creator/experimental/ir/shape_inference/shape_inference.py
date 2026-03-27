from collections.abc import Callable
from typing import cast

import elasticai.creator.function_dispatch as FD
from elasticai.creator import ir

type DataGraph = ir.DataGraph[ir.Node, ir.Edge]
type Registry = ir.Registry[DataGraph]
type TypeHandler = Callable[
    [DataGraph, tuple[tuple[int, ...], ...]], tuple[int, ...]
]


class IrShapeInference:
    def __init__(self) -> None:
        self._ir_factory = ir.DefaultIrFactory()
        self._registry: Registry = ir.Registry()
        self._root = self._ir_factory.graph(ir.attribute(type="module"))

    @FD.dispatch_method(str)
    def _extractors(
        self,
        fn: TypeHandler,
        dg_node: DataGraph,
        input_shapes: tuple[tuple[int, ...], ...],
    ) -> tuple[int, ...]:
        return fn(dg_node, input_shapes)

    @_extractors.key_from_args
    def _get_type_from_dg_node(
        self, dg_node: DataGraph, input_shapes: tuple[tuple[int, ...], ...]
    ) -> str:
        return cast(str, dg_node.attributes["type"])

    @staticmethod
    def _check_and_get_name(name: str | None, fn: TypeHandler) -> str:
        if name is None:
            if hasattr(fn, "__name__") and isinstance(fn.__name__, str):
                return fn.__name__
            else:
                raise TypeError(
                    "specify the type handler's type explicitly if you want to register a non-function callable"
                )
        return name

    @FD.registrar_method
    def register(self, key: str | None, fn: TypeHandler) -> TypeHandler:
        key = self._check_and_get_name(key, fn)
        self._extractors.register(key, fn)
        return fn

    @staticmethod
    def _add_shapes_to_input_edges(
        root: DataGraph, shapes: dict[str, tuple[int, ...]]
    ) -> tuple[DataGraph, list[ir.Node]]:
        starting_nodes = []
        for name, edge in root.edges.items():
            src_node = root.nodes.get(edge.src)
            if src_node is not None:
                if src_node.type == "input":
                    starting_nodes.append(root.nodes[edge.dst])
                    if edge.src not in shapes.keys():
                        raise Exception(f"{edge.src} needs input shape")
                    for start_node, shape in shapes.items():
                        if edge.src == start_node:
                            root = root.add_edge(
                                edge.src, edge.dst, ir.attribute(shape=shape)
                            )
        return root, starting_nodes

    def _get_output_nodes(self, root: DataGraph) -> list[ir.Node]:
        output_nodes = []
        for name, edge in root.edges.items():
            node = root.nodes.get(edge.dst)
            if node is not None:
                if node.type == "output":
                    output_nodes.append(root.nodes[edge.dst])
        return output_nodes

    def _get_shape_for_output_node(
        self, root: DataGraph, reg: Registry, output_node: ir.Node
    ) -> tuple[DataGraph, tuple[int, ...]]:

        def get_all_srcs() -> list[str]:
            srcs = []
            for name, edge in root.edges.items():
                if edge.dst == output_node.name:
                    srcs.append(edge.src)
            return srcs

        def existing_shape_at_edge(src: str, dst: str) -> tuple[int, ...] | None:
            for name, edge in root.edges.items():
                if (edge.src == src) and (edge.dst == dst):
                    if "shape" in edge.attributes.keys():
                        return cast(tuple[int, ...], edge.attributes.get("shape"))
            return None

        incomming_shapes: list[tuple[int, ...]] = []
        for src in get_all_srcs():
            shape = existing_shape_at_edge(src, output_node.name)
            if shape is None:
                new_output_node = root.nodes.get(src)
                if new_output_node is None:
                    raise Exception(f"{src} not found in root.nodes")
                root, resolved_shape = self._get_shape_for_output_node(
                    root, reg, new_output_node
                )
                incomming_shapes.append(resolved_shape)
            else:
                incomming_shapes.append(shape)

        impl = str(output_node.attributes.get("implementation"))
        if impl == "output":
            if not incomming_shapes:
                raise Exception(f"output node {output_node.name} has no incoming edges")
            return root, incomming_shapes[-1]

        dg_node = reg.get(impl)
        if dg_node is None:
            raise Exception(f"{impl} not found in root.nodes")
        outgoing_shape = self._extractors(dg_node, tuple(incomming_shapes))
        if outgoing_shape is None:
            raise Exception(
                f"Shape Calculation Error for {dg_node.attributes['type']} with incomming shapes {incomming_shapes}. The result is {outgoing_shape}"
            )

        for name, edge in root.edges.items():
            if edge.src == output_node.name:
                root = root.add_edge(
                    edge.src, edge.dst, ir.attribute(shape=outgoing_shape)
                )

        return root, outgoing_shape

    def _get_shapes(
        self, root: DataGraph, reg: Registry, output_nodes: list[ir.Node]
    ) -> DataGraph:
        for output_node in output_nodes:
            root, _ = self._get_shape_for_output_node(root, reg, output_node)
        return root

    def __call__(
        self, root: DataGraph, reg: Registry, shapes: dict[str, tuple[int, ...]]
    ) -> DataGraph:

        for root_node, shape in shapes.items():
            node = root.nodes.get(root_node)
            if not isinstance(node, ir.Node):
                raise Exception(f"{root_node} is not a node")
            if node.type != "input":
                raise Exception(f"{root_node} is not a input")
        root, starting_nodes = self._add_shapes_to_input_edges(root, shapes)
        output_nodes = self._get_output_nodes(root)
        new_root = self._get_shapes(root, reg, output_nodes)
        return new_root
