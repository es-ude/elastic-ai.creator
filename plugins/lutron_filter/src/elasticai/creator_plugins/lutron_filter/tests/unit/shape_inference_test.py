"""Unit tests for shape inference rules."""

from collections.abc import Callable

import pytest
from elasticai.creator_plugins.lutron_filter.rules import _ir as ir
from elasticai.creator_plugins.lutron_filter.rules.shape_inference import (
    AttachFilterParametersRule as AttachFilterParameters,
)
from elasticai.creator_plugins.lutron_filter.rules.shape_inference import (
    InferMaxPool1dInChannelsRule as InferMaxPool1dInChannels,
)
from elasticai.creator_plugins.lutron_filter.rules.shape_inference import (
    InferNodeShapesRule,
)

from elasticai.creator.ir2vhdl import (
    Shape,
)
from elasticai.creator_plugins.grouped_filter import FilterParameters


class ObjectUnderTestFactory:
    """Factory for creating test data for shape inference rules."""

    def __init__(self):
        self._created = {}

    def input_node(self, name: str = "input") -> ir.Node:
        return ir.node(name, "input")

    def output_node(self, name: str = "output") -> ir.Node:
        return ir.node(name, "output")

    def conv1d_node(
        self,
        name: str,
        implementation: str | None = None,
        kernel_size: int = 3,
        in_channels: int = 1,
        out_channels: int = 1,
        groups: int = 1,
        stride: int = 1,
    ) -> ir.Node:
        attrs = ir.attribute(
            type="conv1d",
            kernel_size=kernel_size,
            in_channels=in_channels,
            out_channels=out_channels,
            groups=groups,
            stride=stride,
        )
        if implementation:
            attrs = attrs | ir.attribute(implementation=implementation)
        return ir.ir_factory.node(name, attrs)

    def filter_node(
        self,
        name: str,
        implementation: str | None = None,
        kernel_size: int = 1,
        in_channels: int = 1,
        out_channels: int = 1,
        stride: int = 1,
    ) -> ir.Node:
        attrs = ir.attribute(
            type="filter",
            kernel_size=kernel_size,
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
        )
        if implementation:
            attrs = attrs | ir.attribute(implementation=implementation)
        return ir.ir_factory.node(name, attrs)

    def flatten_node(self, name: str) -> ir.Node:
        return ir.node(name, "flatten")

    def maxpool1d_node(
        self,
        name: str,
        kernel_size: int = 2,
        stride: int = 2,
        in_channels: int | None = None,
    ) -> ir.Node:
        attrs = ir.attribute(type="maxpool1d", kernel_size=kernel_size, stride=stride)
        if in_channels is not None:
            attrs = attrs | ir.attribute(in_channels=in_channels)
        return ir.ir_factory.node(name, attrs)

    def batchnorm1d_node(
        self,
        name: str,
        num_features: int,
        implementation: str | None = None,
    ) -> ir.Node:
        attrs = ir.attribute(type="batchnorm1d", num_features=num_features)
        if implementation:
            attrs = attrs | ir.attribute(implementation=implementation)
        return ir.ir_factory.node(name, attrs)

    def binarize_node(self, name: str = "binarize") -> ir.Node:
        return ir.node(name, "binarize")

    def linear_node(
        self,
        name: str,
        implementation: str | None = None,
        in_features: int = 1,
        out_features: int = 1,
    ) -> ir.Node:
        attrs = ir.attribute(
            type="linear",
            in_features=in_features,
            out_features=out_features,
        )
        if implementation:
            attrs = attrs | ir.attribute(implementation=implementation)
        return ir.ir_factory.node(name, attrs)

    def sequential_graph(self, *nodes: ir.Node):
        """Create a sequential graph from a list of nodes."""
        g = ir.ir_factory.graph()
        edges = []
        for src, dst in zip(nodes[:-1], nodes[1:]):
            edges.append((src.name, dst.name))
        graph = g.add_nodes(*nodes).add_edges(*edges)
        registry = ir.Registry()
        return graph, registry

    def with_registry(self, graph: ir.DataGraph, registry: dict):
        """Add implementations to the registry."""
        reg = ir.Registry(
            **{name: ir.ir_factory.graph(attrs) for name, attrs in registry.items()}
        )
        return graph, reg


def _get_shape_from_attrs(attrs, name: str) -> Shape:
    """Helper to extract Shape from attributes."""
    val = attrs.get(name)
    if isinstance(val, Shape):
        return val
    if isinstance(val, tuple):
        return Shape.from_tuple(val)
    return Shape(0, 0)


@pytest.fixture
def make_infer_shape():
    return InferNodeShapesRule


InferRule = InferNodeShapesRule


class TestInferNodeInputOutputShapes:
    """Tests for InferNodeInputOutputShapes rule."""

    def test_sets_input_shape_on_input_node(
        self,
        make_infer_shape: Callable[[Shape], InferRule],
    ):
        """Input node should get the initial shape set."""
        factory = ObjectUnderTestFactory()
        input_node = factory.input_node()
        output_node = factory.output_node()

        graph, _ = factory.sequential_graph(input_node, output_node)

        expected_shape = Shape(16, 100)
        rule = make_infer_shape(Shape(16, 100))
        result_graph = rule(graph)

        shaped_input_node = result_graph.nodes["input"]
        assert expected_shape == shaped_input_node.input_shape
        assert expected_shape == shaped_input_node.output_shape

    def test_sets_output_shape_on_output_node(
        self, make_infer_shape: Callable[[Shape], InferRule]
    ):
        """Output node should get the input_shape propagated to output_shape."""
        factory = ObjectUnderTestFactory()

        graph, _ = factory.sequential_graph(factory.input_node(), factory.output_node())
        expected_shape = Shape(16, 100)
        rule = make_infer_shape(expected_shape)
        result_graph = rule(graph)
        out_node = result_graph.nodes["output"]
        assert (expected_shape, expected_shape) == (
            out_node.input_shape,
            out_node.output_shape,
        )

    def test_infers_flatten_output_shape(
        self, make_infer_shape: Callable[[Shape], InferRule]
    ):
        """Flatten node should produce shape with flattened depth."""
        factory = ObjectUnderTestFactory()
        start_shape = Shape(16, 100)
        graph, _ = factory.sequential_graph(
            factory.input_node(), factory.flatten_node("flatten"), factory.output_node()
        )

        rule = make_infer_shape(start_shape)
        result_graph = rule(graph)

        flatten_node = result_graph.nodes["flatten"]
        assert flatten_node.output_shape == Shape(1, 1600)

    def test_infers_maxpool_output_shape(
        self, make_infer_shape: Callable[[Shape], InferRule]
    ):
        """MaxPool node should compute output shape based on kernel and stride."""
        factory = ObjectUnderTestFactory()
        input_node = factory.input_node()
        maxpool_node = factory.maxpool1d_node("maxpool", kernel_size=2, stride=2)
        output_node = factory.output_node()

        graph, _ = factory.sequential_graph(input_node, maxpool_node, output_node)

        rule = make_infer_shape(Shape(16, 100))
        result_graph = rule(graph)

        result_maxpool = result_graph.nodes["maxpool"]
        # Output shape: depth=16 (unchanged), width=(100-2)/2+1=50

        assert result_maxpool.output_shape == Shape(16, 50)


class TestAttachFilterParameters:
    """Tests for AttachFilterParameters rule."""

    def test_attaches_filter_parameters_from_implementation(self):
        """Nodes should get filter_parameters from their implementations."""
        factory = ObjectUnderTestFactory()

        # Create nodes with implementations
        conv_attrs = ir.attribute(
            type="conv1d",
            kernel_size=3,
            in_channels=16,
            out_channels=32,
            implementation="conv_impl",
        )
        conv_node = ir.ir_factory.node("conv1d", conv_attrs)

        input_node = factory.input_node()
        output_node = factory.output_node()

        # Create implementation with filter_parameters
        impl_attrs = ir.attribute(
            type="conv1d",
            filter_parameters=FilterParameters(
                kernel_size=3,
                in_channels=16,
                out_channels=32,
            ).as_dict(),
        )
        impl = ir.ir_factory.graph(impl_attrs)
        registry = ir.Registry(**{"conv_impl": impl})

        graph = (
            ir.ir_factory.graph()
            .add_nodes(input_node, conv_node, output_node)
            .add_edges(("input", "conv1d"), ("conv1d", "output"))
        )

        rule = AttachFilterParameters()
        result_graph, result_registry = rule(graph, registry)

        result_conv = result_graph.nodes["conv1d"]
        assert "filter_parameters" in result_conv.attributes
        params = result_conv.attributes["filter_parameters"]
        assert params["kernel_size"] == 3
        assert params["in_channels"] == 16
        assert params["out_channels"] == 32

    def test_does_not_modify_input_or_output_nodes(self):
        """Input and output nodes should be skipped."""
        factory = ObjectUnderTestFactory()
        input_node = factory.input_node()
        output_node = factory.output_node()

        graph, _ = factory.sequential_graph(input_node, output_node)

        rule = AttachFilterParameters()
        result_graph, _ = rule(graph, ir.Registry())

        # Should not add filter_parameters to input/output
        assert "filter_parameters" not in result_graph.nodes["input"].attributes
        assert "filter_parameters" not in result_graph.nodes["output"].attributes


class TestInferMaxPool1dInChannels:
    """Tests for InferMaxPool1dInChannels rule."""

    def test_infers_channels_from_upstream_conv(self):
        """MaxPool should get in_channels from upstream conv's out_channels."""
        factory = ObjectUnderTestFactory()
        input_node = factory.input_node()

        conv_attrs = ir.attribute(
            type="conv1d",
            out_channels=32,
            in_channels=16,
            kernel_size=3,
        )
        conv_node = ir.ir_factory.node("conv", conv_attrs)

        maxpool_node = factory.maxpool1d_node("maxpool", in_channels=None)
        output_node = factory.output_node()

        graph = (
            ir.ir_factory.graph()
            .add_nodes(input_node, conv_node, maxpool_node, output_node)
            .add_edges(
                ("input", "conv"),
                ("conv", "maxpool"),
                ("maxpool", "output"),
            )
        )
        registry = ir.Registry()

        rule = InferMaxPool1dInChannels()
        result_graph, _ = rule(graph, registry)

        result_maxpool = result_graph.nodes["maxpool"]
        assert "in_channels" in result_maxpool.attributes
        assert result_maxpool.attributes["in_channels"] == 32

    def test_infers_channels_from_downstream_layer(self):
        """MaxPool should get in_channels from downstream layer's in_channels."""
        factory = ObjectUnderTestFactory()
        input_node = factory.input_node()

        maxpool_node = factory.maxpool1d_node("maxpool", in_channels=None)

        linear_attrs = ir.attribute(type="linear", in_channels=64, out_features=10)
        linear_node = ir.ir_factory.node("linear", linear_attrs)

        output_node = factory.output_node()

        graph = (
            ir.ir_factory.graph()
            .add_nodes(input_node, maxpool_node, linear_node, output_node)
            .add_edges(
                ("input", "maxpool"),
                ("maxpool", "linear"),
                ("linear", "output"),
            )
        )
        registry = ir.Registry()

        rule = InferMaxPool1dInChannels()
        result_graph, _ = rule(graph, registry)

        result_maxpool = result_graph.nodes["maxpool"]
        assert "in_channels" in result_maxpool.attributes
        assert result_maxpool.attributes["in_channels"] == 64

    def test_preserves_existing_in_channels(self):
        """MaxPool with existing in_channels should not be modified."""
        factory = ObjectUnderTestFactory()
        input_node = factory.input_node()

        # MaxPool already has in_channels set
        maxpool_node = factory.maxpool1d_node("maxpool", in_channels=16)

        output_node = factory.output_node()

        graph, _ = factory.sequential_graph(input_node, maxpool_node, output_node)
        registry = ir.Registry()

        rule = InferMaxPool1dInChannels()
        result_graph, _ = rule(graph, registry)

        result_maxpool = result_graph.nodes["maxpool"]
        assert result_maxpool.attributes["in_channels"] == 16

    def test_prefers_upstream_search(self):
        """Should prefer searching upstream over downstream."""
        factory = ObjectUnderTestFactory()
        input_node = factory.input_node()

        # Upstream conv with out_channels=32
        upstream_attrs = ir.attribute(
            type="conv1d", out_channels=32, in_channels=16, kernel_size=3
        )
        upstream_node = ir.ir_factory.node("upstream", upstream_attrs)

        # Downstream linear with in_channels=64
        maxpool_node = factory.maxpool1d_node("maxpool", in_channels=None)
        downstream_attrs = ir.attribute(type="linear", in_channels=64, out_features=10)
        downstream_node = ir.ir_factory.node("downstream", downstream_attrs)

        output_node = factory.output_node()

        graph = (
            ir.ir_factory.graph()
            .add_nodes(
                input_node, upstream_node, maxpool_node, downstream_node, output_node
            )
            .add_edges(
                ("input", "upstream"),
                ("upstream", "maxpool"),
                ("maxpool", "downstream"),
                ("downstream", "output"),
            )
        )
        registry = ir.Registry()

        rule = InferMaxPool1dInChannels()
        result_graph, _ = rule(graph, registry)

        result_maxpool = result_graph.nodes["maxpool"]
        # Should prefer upstream value (32) over downstream (64)
        assert result_maxpool.attributes["in_channels"] == 32

    def test_raises_when_channels_not_found(self):
        """Should raise ValueError when channel info cannot be found."""
        factory = ObjectUnderTestFactory()
        input_node = factory.input_node()
        maxpool_node = factory.maxpool1d_node("maxpool", in_channels=None)
        output_node = factory.output_node()

        # Create isolated chain where channels can't be found
        graph, _ = factory.sequential_graph(input_node, maxpool_node, output_node)
        registry = ir.Registry()

        rule = InferMaxPool1dInChannels()
        with pytest.raises(ValueError, match="Could not find"):
            rule(graph, registry)
