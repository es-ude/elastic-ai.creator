"""Shape inference rules for LUTron filter graphs.

This module provides rules for inferring input/output shapes, attaching filter
parameters, and propagating channel information through the graph.
"""

from collections.abc import Callable
from typing import Any

import elasticai.creator.ir as ir
from elasticai.creator.experimental.ir.shape_inference import (
    shapes_calculation_functions as shape_calc,
)
from elasticai.creator.experimental.ir.shape_inference.shape_inference import (
    Node as _Node,
)
from elasticai.creator.experimental.ir.shape_inference.shapes_calculation_functions import (
    flatten_output_shape,
    maxpool1d_output_shape,
)
from elasticai.creator.hdl_ir import DataGraph as VhdDataGraph
from elasticai.creator.hdl_ir import Node as VhdNode
from elasticai.creator.hdl_ir import Shape
from elasticai.creator.ir.executor import DataGraph as _TypedDGraph
from elasticai.creator.ir2vhdl import factory as vhdl_factory
from elasticai.creator_plugins.grouped_filter import FilterParameters

from ._ir import DataGraph, Node, Registry


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
        self._reg = registry
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
            pred_impl = self._reg[pred_node.implementation]

            # Check the node's own attributes first
            out_ch = pred_impl.attributes.get("out_channels")
            if out_ch is not None:
                val = _safe_int(out_ch)
                if val > 0:
                    return val

            num_feat = pred_impl.attributes.get("num_features")
            if num_feat is not None:
                val = _safe_int(num_feat)
                if val > 0:
                    return val

            if "output_shape" in pred_impl.attributes:
                out_shape = pred_impl.attributes["output_shape"]
                if len(out_shape) > 1:
                    val = _safe_int(out_shape[0])
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
            succ_impl = self._reg[succ_node.implementation]

            # Check the node's own attributes first
            in_ch = succ_impl.attributes.get("in_channels")
            if in_ch is not None:
                val = _safe_int(in_ch)
                if val > 0:
                    return val

            num_feat = succ_impl.attributes.get("num_features")
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

    if isinstance(fp_data, FilterParameters):  # zuban: ignore[unreachable]
        return fp_data

    if isinstance(fp_data, dict):  # zuban: ignore[unreachable]
        return FilterParameters.from_dict(fp_data)  # zuban: ignore[unreachable]

    return None


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


def _unpack_input_shape[*T](input_shapes: tuple[tuple[*T], ...]) -> tuple[*T]:
    if len(input_shapes) != 1:
        raise ValueError(
            "invalid input_shapes, expect 1-tuple, for operation taking single argument"
        )
    return input_shapes[0]


def _unwrap_scalar(shape: tuple[int, ...] | int) -> int:
    if isinstance(shape, tuple):
        if len(shape) > 1 or len(shape) == 0:
            raise ValueError()
        return shape[0]
    return shape


def create_shape_inference() -> Callable[
    [_TypedDGraph[Node, ir.Edge], Shape, int], VhdDataGraph
]:
    infer: ir.ExecutionOrderGraphReducer[Node, VhdDataGraph] = (
        ir.ExecutionOrderGraphReducer()
    )

    def get_pred(n: Node, acc: VhdDataGraph) -> VhdNode:
        preds = list(acc.predecessors[n.name])
        if len(preds) > 1:
            raise ValueError("unsupported join of multiple data lines")
        return acc.nodes[preds[0]]

    @infer.register()  # ty:ignore[invalid-argument-type]
    def filter(n: Node, acc: VhdDataGraph) -> VhdDataGraph:
        preds = list(acc.predecessors[n.name])
        if len(preds) > 1:
            raise ValueError("unsupported join of multiple data lines")
        elif len(preds) == 0:
            return acc
        pred = acc.nodes[preds[0]]
        params = FilterParameters.from_dict(n.attributes["filter_parameters"])
        acc = acc.add_node(
            n.name,
            n.attributes
            | params.as_dict()
            | {
                "input_shape": pred.output_shape.to_tuple(),
                "output_shape": (params.out_channels, params.num_steps),
            },
        )
        return acc

    @infer.register()  # ty:ignore[invalid-argument-type]
    def output(n: Node, acc: VhdDataGraph) -> VhdDataGraph:
        pred = get_pred(n, acc)
        return acc.add_node(
            n.name,
            n.attributes
            | {
                "input_shape": pred.input_shape.to_tuple(),
                "output_shape": pred.output_shape.to_tuple(),
            },
        )

    @infer.register()  # ty:ignore[invalid-argument-type]
    def input(n: Node, acc: VhdDataGraph) -> VhdDataGraph:
        return acc

    def _do_infer(
        g: _TypedDGraph[Node, ir.Edge], input_shape: Shape, num_input_bits: int
    ) -> VhdDataGraph:
        input_nodes: list[Node] = []
        for n in g.nodes.values():
            if n.type == "input":
                input_nodes.append(n)

        input_shape = Shape(num_input_bits * input_shape.depth, input_shape.width)
        if len(input_nodes) != 1:
            raise ValueError("supporting only a single input node")
        input = input_nodes[0]
        return infer(
            g,
            vhdl_factory.graph_from_other(g).add_node(
                input.name,
                input.attributes
                | {
                    "input_shape": input_shape.to_tuple(),
                    "output_shape": input_shape.to_tuple(),
                },
            ),
        )

    return _do_infer


def _maxpool1d_shape(
    graph: ir.DataGraph[ir.Node, ir.Edge],
    input_shapes: tuple[tuple[int, ...], ...],
) -> tuple[int, ...]:
    attr = graph.attributes
    kernel_size = _unwrap_scalar(attr["kernel_size"])
    return shape_calc.maxpool1d_output_shape(
        x_shape=_unpack_input_shape(input_shapes),  # type: ignore[arg-type]
        kernel_size=kernel_size,
        stride=attr.get_int("stride", kernel_size),
        padding=attr.get_int("padding", 0),
        dilation=attr.get_int("dilation", 1),
    )


def _conv1d_shape(
    graph: ir.DataGraph[ir.Node, ir.Edge], input_shapes: tuple[tuple[int, ...], ...]
) -> tuple[int, ...]:
    attr = graph.attributes
    kernel_size = _unwrap_scalar(attr["kernel_size"])
    if len(input_shapes) != 3:
        raise ValueError()
    return shape_calc.conv1d_output_shape(
        x_shape=_unpack_input_shape(input_shapes),  # type: ignore[arg-type]
        out_channels=_unwrap_scalar(attr["output_channels"]),
        kernel_size=kernel_size,
        stride=attr.get_int("stride", kernel_size),
        padding=attr.get_int("padding", 0),
        dilation=attr.get_int("dilation", 1),
    )


def flatten(_, input_shapes: tuple[tuple[int, ...], ...]) -> tuple[int, ...]:
    input_shape = _unpack_input_shape(input_shapes)
    return flatten_output_shape(tuple((1, *input_shape)))


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
    if isinstance(input_shape, Shape):  # zuban: ignore[unreachable]
        return input_shape
    if isinstance(input_shape, tuple):
        return Shape.from_tuple(input_shape)  # type: ignore
    return Shape(0, 0)


def _get_output_shape(graph: DataGraph, node_name: str) -> Shape:
    """Get the output shape for a node."""
    attrs = graph.nodes[node_name].attributes
    output_shape = attrs.get("output_shape")
    if isinstance(output_shape, Shape):  # zuban: ignore[unreachable]
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
