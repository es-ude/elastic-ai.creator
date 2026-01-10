import elasticai.creator.ir.ir_v2 as ir
from elasticai.creator.ir2vhdl import Shape
from elasticai.creator.ir2vhdl import factory as _factory
from elasticai.creator_plugins.grouped_filter import FilterParameters


def network(
    input_shape: tuple[int, ...], kernel_size: int, out_channels: int, stride: int = 2
):
    out_length = (input_shape[1] - kernel_size) // stride + 1
    in_channels = input_shape[0]
    _net = (
        _factory.graph(type="network")
        .add_nodes(
            _factory.node(
                "input",
                input_shape=Shape(*input_shape),
                output_shape=Shape(*input_shape),
                type="input",
                implementation="",
            ),
            _factory.node(
                "conv0_i0",
                ir.attribute(
                    dict(
                        filter_parameters=FilterParameters(
                            kernel_size=kernel_size,
                            in_channels=in_channels,
                            out_channels=out_channels,
                            stride=stride,
                        ).as_dict(),
                    ),
                    type="filter",
                    implementation="conv0",
                    input_shape=input_shape,
                    output_shape=(out_channels, out_length),
                ),
            ),
            _factory.node(
                "conv1_i0",
                ir.attribute(
                    dict(
                        filter_parameters=FilterParameters(
                            kernel_size=kernel_size,
                            in_channels=out_channels,
                            out_channels=out_channels,
                            stride=1,
                        ).as_dict()
                    ),
                    type="filter",
                    implementation="conv1",
                    input_shape=(out_channels, out_length),
                    output_shape=(out_channels, out_length - kernel_size + 1),
                ),
            ),
            _factory.node(
                "output",
                type="output",
                implementation="",
                input_shape=Shape(out_channels, out_length - kernel_size + 1),
                output_shape=Shape(out_channels, out_length - kernel_size + 1),
            ),
        )
        .add_edges(
            _factory.edge("input", "conv0_i0", src_dst_indices=tuple()),
            _factory.edge("conv0_i0", "conv1_i0", src_dst_indices=tuple()),
            _factory.edge("conv1_i0", "output", src_dst_indices=tuple()),
        )
    )

    return _net
