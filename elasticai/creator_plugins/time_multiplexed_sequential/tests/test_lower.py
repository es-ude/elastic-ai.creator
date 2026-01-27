from collections.abc import Callable

import pytest

import elasticai.creator.function_dispatch as FD
from elasticai.creator import ir
from elasticai.creator.ir2vhdl import DataGraph, Shape, factory
from elasticai.creator_plugins.grouped_filter import FilterParameters
from elasticai.creator_plugins.time_multiplexed_sequential.src import (
    network as _network_handler,
)
from elasticai.creator_plugins.time_multiplexed_sequential.src import (
    sequential,
)

from .high_level_network import network


@pytest.fixture
def lower():
    def get_key(graph: DataGraph, registry: ir.Registry) -> str:
        return graph.type

    def get_key_from_fn(
        fn: Callable[[DataGraph, ir.Registry], tuple[DataGraph, ir.Registry]],
    ) -> str:
        return fn.__name__  # type: ignore

    @FD.create_keyed_dispatch(get_key, get_key_from_fn)
    def _lower(
        fn: Callable[[DataGraph, ir.Registry], tuple[DataGraph, ir.Registry]],
        graph: DataGraph,
        registry: ir.Registry,
    ) -> tuple[DataGraph, ir.Registry]:
        return fn(graph, registry)

    _lower.register()(sequential)
    _lower.register()(_network_handler)
    return _lower


def vhdl_node(name, type, implementation, input_shape, output_shape, attributes=None):
    if attributes is not None:
        attributes = ir.attribute(**attributes)
    else:
        attributes = ir.attribute()
    return factory.node(
        name,
        attributes,
        type=type,
        implementation=implementation,
        input_shape=input_shape,
        output_shape=output_shape,
    )


def edge(src, dst, src_dst_indices):
    return factory.edge(src, dst, src_dst_indices=src_dst_indices)


def test_sets_input_correctly_for_version_without_sliding_window(lower):
    lowered, _ = lower(
        network(
            input_shape=(1, 3),
            kernel_size=3,
            out_channels=1,
        ),
        ir.Registry(),
    )
    expected_input_node = vhdl_node(
        name="input",
        type="input",
        implementation="",
        input_shape=Shape(1, 3),
        output_shape=Shape(1, 3),
    )
    assert lowered.nodes["input"] == expected_input_node


def test_first_shift_reg_has_three_output_wires():
    in_channels = 2
    length = 4
    _net = network(
        input_shape=(in_channels, length), kernel_size=2, out_channels=2, stride=1
    )

    lowered, _ = sequential(_net, ir.Registry())
    node = lowered.nodes["shift_register_i0"]
    assert node.output_shape == (2, 2)


def test_shift_reg_predecessor_is_conv1():
    in_channels = 2
    length = 4
    _net = network(
        input_shape=(in_channels, length), kernel_size=2, out_channels=2, stride=1
    )

    lowered, _ = sequential(_net, ir.Registry())
    p = set(iter(lowered.successors["shift_register_i0"].keys()))

    assert p == {"conv1_i0"}


def test_second_conv_has_three_input_wires():
    kernel_size = 1
    _net = network(
        input_shape=(3, 2), kernel_size=kernel_size, out_channels=3, stride=1
    )
    hl_conv1 = _net.nodes["conv1_i0"]
    lowered, _ = sequential(_net, ir.Registry())
    low_level_conv1 = lowered.nodes["conv1_i0"]
    expected_shape = Shape(hl_conv1.input_shape.depth, kernel_size)
    assert low_level_conv1.input_shape == expected_shape


class TestsForSecondConv:
    def network(self):
        return network(input_shape=(3, 2), kernel_size=1, out_channels=3, stride=1)

    def get_hl_conv(self):
        pass

    def test_high_and_low_level_have_same_in_channels(self):
        _net = self.network()
        hl_conv1 = _net.nodes["conv1_i0"]
        lowered, _ = sequential(_net, ir.Registry())
        low_level_conv1 = lowered.nodes["conv1_i0"]
        assert low_level_conv1.input_shape.depth == hl_conv1.input_shape.depth


def test_add_correct_edges(lower):
    net = network(
        input_shape=(2, 7),
        kernel_size=3,
        out_channels=1,
    )
    lowered = tuple(lower(net, ir.Registry()))[0]
    edges = tuple(lowered.edges.values())
    expected = (
        edge(
            src="input",
            dst="conv0_i0",
            src_dst_indices=tuple(),
        ),
        edge(
            src="conv0_i0",
            dst="striding_shift_register_i0",
            src_dst_indices=tuple(),
        ),
        edge(
            src="striding_shift_register_i0",
            dst="conv1_i0",
            src_dst_indices=tuple(),
        ),
        edge(src="conv1_i0", dst="output", src_dst_indices=tuple()),
    )
    assert edges == expected


def test_remember_stride_after_pointwise_filter() -> None:
    in_channels = 1
    input_shape = Shape(in_channels, 20)

    net = factory.graph(type="network").add_nodes(
        vhdl_node(
            name="input",
            input_shape=input_shape,
            output_shape=input_shape,
            type="input",
            implementation="",
        ),
        vhdl_node(
            name="f0_i0",
            type="filter",
            implementation="f0",
            input_shape=Shape(1, 20),
            output_shape=Shape(2, 10),
            attributes=dict(
                filter_parameters=FilterParameters(
                    kernel_size=2, in_channels=1, out_channels=2, stride=2
                ).as_dict()
            ),
        ),
        vhdl_node(
            name="f1_i0",
            type="filter",
            implementation="f1",
            input_shape=Shape(2, 10),
            output_shape=Shape(3, 10),
            attributes=dict(
                filter_parameters=FilterParameters(
                    kernel_size=1, out_channels=3, in_channels=2
                ).as_dict()
            ),
        ),
        vhdl_node(
            name="f2_i0",
            type="filter",
            implementation="f2",
            input_shape=Shape(3, 10),
            output_shape=Shape(2, 9),
            attributes=dict(
                filter_parameters=FilterParameters(
                    kernel_size=2, out_channels=2, in_channels=3
                ).as_dict()
            ),
        ),
        vhdl_node(
            name="output",
            type="output",
            implementation="",
            input_shape=Shape(2, 9),
            output_shape=Shape(2, 9),
        ),
    )
    edge_sequence = ["input", "f0_i0", "f1_i0", "f2_i0", "output"]
    net = net.add_edges(
        *(
            edge(src, dst, tuple())
            for src, dst in zip(edge_sequence[:-1], edge_sequence[1:])
        )
    )
    lowered, _ = sequential(net, ir.Registry())
    follower_of_pointwise = lowered.nodes[list(lowered.successors["f1_i0"].keys())[0]]
    assert follower_of_pointwise.type == "striding_shift_register"
