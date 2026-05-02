"""Shape inference rules for LUTron filter graphs.

This module provides rules for inferring input/output shapes, attaching filter
parameters, and propagating channel information through the graph.
"""

from typing import Any

from elasticai.creator.experimental.ir.shape_inference import (
    get_default_shape_inference,
)
from elasticai.creator.experimental.ir.shape_inference.shape_inference import (
    Node as _Node,
)
from elasticai.creator.experimental.ir.shape_inference.shapes_calculation_functions import (
    flatten_output_shape,
    maxpool1d_output_shape,
)
from elasticai.creator.ir2vhdl import DataGraph as VhdDataGraph
from elasticai.creator.ir2vhdl import Shape
from elasticai.creator.ir2vhdl import factory as vhdl_factory
from elasticai.creator_plugins.grouped_filter import FilterParameters

from ._ir import DataGraph, Node, Registry


def _get_predecessors(graph: DataGraph, node_name: str) -> list[str]:
    """Get list of predecessor node names for a given node."""
    preds = graph.predecessors.get(node_name, {})
    return list(preds.keys())


def _get_successors(graph: DataGraph, node_name: str) -> list[str]:
    """Get list of successor node names for a given node."""
    succs = graph.successors.get(node_name, {})
    return list(succs.keys())


def _get_input_shape(graph: DataGraph, node_name: str) -> Shape:
    """Get the input shape for a node."""
    attrs = graph.nodes[node_name].attributes
    input_shape = attrs.get("input_shape")
    if isinstance(input_shape, Shape):
        return input_shape
    if isinstance(input_shape, tuple):
        return Shape.from_tuple(input_shape)  # type: ignore
    return Shape(0, 0)


def _get_output_shape(graph: DataGraph, node_name: str) -> Shape:
    """Get the output shape for a node."""
    attrs = graph.nodes[node_name].attributes
    output_shape = attrs.get("output_shape")
    if isinstance(output_shape, Shape):
        return output_shape
    if isinstance(output_shape, tuple):
        return Shape.from_tuple(output_shape)  # type: ignore
    return Shape(0, 0)


def _safe_int(value: Any, default: int = 0) -> int:
    """Safely convert a value to int."""
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return default
    return default


def _safe_str(value: Any, default: str = "") -> str:
    """Safely convert a value to str."""
    if value is None:
        return default
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        return str(value)
    return default


class AttachFilterParametersRule:
    """Rule that attaches filter_parameters from implementations to nodes.

    This rule looks up the implementation for each node and copies the
    filter_parameters from the implementation to the node.
    """

    name: str = "AttachFilterParameters"

    def __call__(
        self, graph: DataGraph, registry: Registry
    ) -> tuple[DataGraph, Registry]:
        for node in graph.nodes.values():
            if node.type in ("input", "output"):
                continue

            impl_name = _safe_str(node.attributes.get("implementation"))
            if impl_name and impl_name in registry:
                impl = registry[impl_name]
                if "filter_parameters" in impl.attributes:
                    new_attrs = node.attributes | {
                        "filter_parameters": impl.attributes["filter_parameters"]
                    }
                    graph = graph.add_node(node.name, new_attrs)

        return graph, registry


class InferMaxPool1dInChannelsRule:
    """Rule that infers in_channels for maxpool1d nodes.

    This rule searches upstream and downstream through nodes with single
    connections to find channel information.
    """

    name: str = "InferMaxPool1dInChannels"

    def __call__(
        self, graph: DataGraph, registry: Registry
    ) -> tuple[DataGraph, Registry]:
        # Find maxpool nodes that need channel inference
        for node in graph.nodes.values():
            if node.type == "maxpool1d" and "in_channels" not in node.attributes:
                channels = self._find_channels(graph, node.name)
                if channels == 0:
                    raise ValueError(
                        f"Could not find 'in_channels' or 'out_channels' for node {node.name}"
                    )
                new_attrs = node.attributes | {"in_channels": channels}
                graph = graph.add_node(node.name, new_attrs)

        return graph, registry

    def _find_channels(self, graph: DataGraph, node_name: str) -> int:
        """Find channel count by searching upstream or downstream."""
        # First try searching for out_channels upstream
        channels = self._search_upstream(graph, node_name)
        if channels > 0:
            return channels

        # Then try searching for in_channels downstream
        return self._search_downstream(graph, node_name)

    def _search_upstream(self, graph: DataGraph, node_name: str) -> int:
        """Search upstream through the graph for out_channels or num_features."""
        node = graph.nodes[node_name]

        while True:
            preds = _get_predecessors(graph, node.name)
            if len(preds) != 1:
                break
            pred_name = preds[0]
            pred_node = graph.nodes[pred_name]

            # Check the node's own attributes first
            out_ch = pred_node.attributes.get("out_channels")
            if out_ch is not None:
                val = _safe_int(out_ch)
                if val > 0:
                    return val

            num_feat = pred_node.attributes.get("num_features")
            if num_feat is not None:
                val = _safe_int(num_feat)
                if val > 0:
                    return val

            # Move upstream
            node = pred_node

        return 0

    def _search_downstream(self, graph: DataGraph, node_name: str) -> int:
        """Search downstream through the graph for in_channels or num_features."""
        node = graph.nodes[node_name]

        while True:
            succs = _get_successors(graph, node.name)
            if len(succs) != 1:
                break
            succ_name = succs[0]
            succ_node = graph.nodes[succ_name]

            # Check the node's own attributes first
            in_ch = succ_node.attributes.get("in_channels")
            if in_ch is not None:
                val = _safe_int(in_ch)
                if val > 0:
                    return val

            num_feat = succ_node.attributes.get("num_features")
            if num_feat is not None:
                val = _safe_int(num_feat)
                if val > 0:
                    return val

            # Move downstream
            node = succ_node

        return 0


def _get_filter_parameters(node: Node) -> FilterParameters | None:
    """Extract FilterParameters from a node's attributes."""
    fp_data = node.attributes.get("filter_parameters")
    if fp_data is None:
        # Also check kernel_size, in_channels, out_channels directly
        attrs = node.attributes
        kernel_size = attrs.get("kernel_size")
        in_channels = attrs.get("in_channels")
        out_channels = attrs.get("out_channels")

        if all(v is not None for v in (kernel_size, in_channels, out_channels)):
            ks = _safe_int(kernel_size)
            ic = _safe_int(in_channels)
            oc = _safe_int(out_channels)
            stride = _safe_int(attrs.get("stride", 1), 1)
            groups = _safe_int(attrs.get("groups", 1), 1)

            if ks > 0 and ic > 0 and oc > 0:
                return FilterParameters(
                    kernel_size=ks,
                    in_channels=ic,
                    out_channels=oc,
                    stride=stride,
                    groups=groups,
                )
        return None

    if isinstance(fp_data, FilterParameters):
        return fp_data

    if isinstance(fp_data, dict):
        return FilterParameters.from_dict(fp_data)

    return None


def _unpack_input_shape(input_shapes: tuple[tuple[int, ...], ...]) -> tuple[int, ...]:
    if len(input_shapes) > 1:
        raise ValueError(
            "invalid input_shapes, expect 1-tuple, for operation taking single argument"
        )
    return input_shapes[0]


def maxpool1d(
    node: _Node, input_shapes: tuple[tuple[int, ...], ...]
) -> tuple[int, ...]:
    input_shape = _unpack_input_shape(input_shapes)
    match input_shape:
        case N, C, L:
            _shape = (N, C, L)
        case C, L:
            _shape = (1, C, L)

    return maxpool1d_output_shape(
        _shape,
        kernel_size=node.attributes["kernel_size"],
        stride=node.attributes["stride"],
    )


def flatten(_, input_shapes: tuple[tuple[int, ...], ...]) -> tuple[int, ...]:
    input_shape = _unpack_input_shape(input_shapes)
    return flatten_output_shape(tuple((1, *input_shape)))


class InferNodeShapesRule:
    def __init__(self, input_shape: Shape):
        self._inferer = get_default_shape_inference()
        self._inferer.register_node("maxpool1d")(maxpool1d)
        self._inferer.register_node()(flatten)
        self._input_shape = input_shape

    @staticmethod
    def _convert_shape_style(core_creator_style: tuple[int, ...]) -> Shape:
        match core_creator_style:
            case _, C, L:
                return Shape(C, L)
            case C, L:
                return Shape(C, L)
            case _:
                raise ValueError(
                    f"Invalid Shape: expected (C, L) got {core_creator_style}"
                )

    def __call__(self, g: DataGraph) -> VhdDataGraph:
        input_node = ""
        inferer = self._inferer
        input_shape = self._input_shape
        for n in g.nodes.values():
            if n.type == "input":
                input_node = n.name
                break
        if input_node == "":
            raise ValueError("no input_node found")
        new_g = inferer(g, Registry(), {input_node: input_shape.to_tuple()})
        g = g.add_node(
            input_node,
            g.nodes[input_node].attributes | dict(input_shape=input_shape.to_tuple()),
        )
        for edge in new_g.edges.values():
            src = g.nodes[edge.src]
            dst = g.nodes[edge.dst]
            shape = self._convert_shape_style(edge.shape).to_tuple()
            src_attrs = dict(output_shape=shape)
            dst_attrs = dict(input_shape=shape)
            if src.type == "input":
                src_attrs = src_attrs | dst_attrs
            if dst.type == "output":
                dst_attrs = src_attrs
            g = g.add_nodes(
                (src.name, src.attributes | src_attrs),
                (dst.name, dst.attributes | dst_attrs),
            )

        return vhdl_factory.graph(other=g)
