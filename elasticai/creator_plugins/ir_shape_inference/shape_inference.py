from sympy.multipledispatch.tests.test_dispatcher import inc
from PIL.Image import new
from lxml.html.builder import DATA
from svgwrite.extensions.shapes import star
from collections.abc import Callable

from elasticai.creator import ir
import elasticai.creator.function_dispatch as FD

type DataGraph = ir.DataGraph[ir.Node, ir.Edge]
type Registry = ir.Registry[DataGraph]
type TypeHandler = Callable[[DataGraph, tuple[int, ...]], tuple[int, ...]]


def serialize(root: DataGraph, reg: Registry):
    serializer = ir.IrSerializerLegacy()
    result = {k: serializer.serialize(reg[k]) for k in reg}
    result[""] = serializer.serialize(root)
    return result


class IrShapeInference:
    def __init__(self) -> None:
        self._ir_factory = ir.DefaultIrFactory()
        self._registry: Registry = ir.Registry()
        self._root = self._ir_factory.graph(ir.attribute(type="module"))

    @FD.dispatch_method(str)
    def _extractors(
        self, fn: TypeHandler, dg_node: DataGraph, shape: tuple[int, ...]
    ) -> tuple[int, ...]:
        return fn(dg_node, shape)

    @_extractors.key_from_args
    def _get_type_from_dg_node(self, dg_node: DataGraph, shape: tuple[int, ...]) -> str:
        return dg_node.attributes["type"]

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
        print(serialize(root, reg))

        def get_all_srcs() -> list[str]:
            srcs = []
            for name, edge in root.edges.items():
                if edge.dst == output_node.name:
                    srcs.append(edge.src)
            return srcs

        def existing_shape_at_edge(src: str, dst: str) -> None | tuple[int, ...]:
            for name, edge in root.edges.items():
                if (edge.src == src) and (edge.dst == dst):
                    if "shape" in edge.attributes.keys():
                        attr = edge.attributes
                        return attr.get("shape")  # ty:ignore[invalid-return-type]

        incomming_shapes = []
        for src in get_all_srcs():
            shape = existing_shape_at_edge(src, output_node.name)
            if shape is None:
                new_output_node = root.nodes.get(src)
                if new_output_node is None:
                    raise Exception(f"{src} not found in root.nodes")
                root, shape = self._get_shape_for_output_node(
                    root, reg, new_output_node
                )
                print(serialize(root, reg))
                incomming_shapes.append(shape)
            else:
                incomming_shapes.append(shape)

        impl = str(output_node.attributes.get("implementation"))
        if impl == "output":
            return root, shape

        dg_node = reg.get(impl)
        if dg_node is None:
            raise Exception(f"{impl} not found in root.nodes")
        # print(f"{incomming_shapes=}")
        outgoing_shape = self._extractors(dg_node, tuple(incomming_shapes))
        if outgoing_shape is None:
            raise Exception(
                f"Shape Calculation Error for {dg_node.attributes['type']} with incomming shapes {incomming_shapes}. The result is {outgoing_shape}"
            )

        # print(f"{output_node.name}")
        # print(f"{outgoing_shape=}")
        for name, edge in root.edges.items():
            if edge.src == output_node.name:
                root = root.add_edge(
                    edge.src, edge.dst, ir.attribute(shape=outgoing_shape)
                )

        return root, outgoing_shape

    def _get_shapes(
        self, root: DataGraph, reg: Registry, output_nodes: list[ir.Node]
    ) -> DataGraph:
        """
        # CARE
        # while construction needed, because otherwise it is not possible to handle multiple input edges properly
        # THE FOLLOWING IS NOT GOOD
        # get all outgoing edges from node
        # for each outgoing shape
        #       get the respective node
        #       if it is type output continue
        #       get the corresponding registry graph
        #       calculate the output shape with registry graph and _extractors
        #       add that output_shape as an attribute to the corresponding edges
        #       create an shapes_dict that you can use to call this function recursively
        """
        out_nodes = []
        for output_node in output_nodes:
            root, _ = self._get_shape_for_output_node(root, reg, output_node)
            print(serialize(root, reg))
        return root

    def __call__(
        self, root: DataGraph, reg: Registry, shapes: dict[str, tuple[int, ...]]
    ) -> DataGraph:

        for k, v in root.nodes.items():
            print(f"{k}: {v}")

        for k, v in root.edges.items():
            print(f"{k}: {v}")

        for root_node, shape in shapes.items():
            node = root.nodes.get(root_node)
            if not isinstance(node, ir.Node):
                raise Exception(f"{root_node} is not a node")
            if node.type != "input":
                raise Exception(f"{root_node} is not a input")
        root, starting_nodes = self._add_shapes_to_input_edges(root, shapes)
        print(serialize(root, reg))
        output_nodes = self._get_output_nodes(root)
        print(output_nodes)
        # print(serialize(root, reg))
        new_root = self._get_shapes(root, reg, output_nodes)
        print(serialize(new_root, reg))
        return new_root
