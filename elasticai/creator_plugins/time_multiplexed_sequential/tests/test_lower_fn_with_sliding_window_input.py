import elasticai.creator.ir.ir_v2 as ir
from elasticai.creator.ir2vhdl import IrFactory, Shape
from elasticai.creator_plugins.grouped_filter import FilterParameters
from elasticai.creator_plugins.time_multiplexed_sequential.src import (
    sequential,
)

from .high_level_network import network

_factory = IrFactory()


def test_inserts_edge_from_input_to_sliding_window() -> None:
    nw = network(
        input_shape=(1, 4),
        kernel_size=3,
        out_channels=1,
        stride=2,
    )
    lowered, _ = sequential(nw, ir.Registry())
    expected = [
        _factory.edge(
            "input",
            "conv0_i0",
            src_dst_indices=tuple(),
        )
    ]
    actual = []
    for e in lowered.edges.values():
        if e.src == "input":
            actual.append(e)

    assert actual == expected


def test_inserts_striding_shift_register():
    kernel_size = 3
    out_channels = 1
    nw = network(
        kernel_size=kernel_size,
        out_channels=out_channels,
        input_shape=(1, 10),
        stride=2,
    )
    lowered, _ = sequential(nw, ir.Registry())
    type = "striding_shift_register"
    name = f"{type}_i0"
    expected = _factory.node(
        name,
        ir.attribute({"generic_map": {"stride": 2}}),
        input_shape=Shape(1, out_channels),
        output_shape=Shape(1, kernel_size),
        implementation=type,
        type=type,
    ).attributes
    assert lowered.nodes[name].attributes == expected


def test_convolution_is_translated_to_unclocked_combinatorial():
    kernel_size = 3
    out_channels = 1
    nw = network(
        kernel_size=kernel_size,
        out_channels=out_channels,
        input_shape=(1, 10),
        stride=2,
    )
    lowered, _ = sequential(nw, ir.Registry())
    assert lowered.nodes["conv0_i0"].type == "unclocked_combinatorial"


def test_inserts_conv_node():
    lowered, _ = sequential(
        network(input_shape=(1, 10), kernel_size=3, out_channels=1, stride=1),
        ir.Registry(),
    )

    expected = _factory.node(
        "sliding_window",
        ir.attribute(
            filter_parameters=FilterParameters(
                kernel_size=3,
                in_channels=1,
                out_channels=1,
                output_size=1,
                stride=1,
            ).as_dict()
        ),
        input_shape=Shape(
            1,
            3,
        ),
        output_shape=Shape(
            1,
            1,
        ),
        type="unclocked_combinatorial",
        implementation="conv0",
    )
    actual = lowered.nodes[tuple(lowered.successors["input"].keys())[0]]
    assert actual.attributes == expected.attributes
