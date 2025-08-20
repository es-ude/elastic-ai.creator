from elasticai.creator.ir2vhdl import Implementation, edge, vhdl_node
from elasticai.creator_plugins.grouped_filter import FilterParameters


def network(input_shape, kernel_size, out_channels, stride=2):
    out_length = (input_shape[1] - kernel_size) // stride + 1
    in_channels = input_shape[0]
    _net = Implementation(name="network", type="network")
    _net.add_nodes(
        (
            vhdl_node(
                name="input",
                input_shape=input_shape,
                output_shape=input_shape,
                type="input",
                implementation="",
            ),
            vhdl_node(
                name="conv0_i0",
                type="filter",
                implementation="conv0",
                input_shape=input_shape,
                output_shape=(out_channels, out_length),
                attributes=dict(
                    filter_parameters=FilterParameters(
                        kernel_size=kernel_size,
                        in_channels=in_channels,
                        out_channels=out_channels,
                        stride=stride,
                    ).as_dict(),
                ),
            ),
            vhdl_node(
                name="conv1_i0",
                type="filter",
                implementation="conv1",
                input_shape=(out_channels, out_length),
                output_shape=(out_channels, out_length - kernel_size + 1),
                attributes=dict(
                    filter_parameters=FilterParameters(
                        kernel_size=kernel_size,
                        in_channels=out_channels,
                        out_channels=out_channels,
                        stride=1,
                    ).as_dict()
                ),
            ),
            vhdl_node(
                name="output",
                type="output",
                implementation="",
                input_shape=(out_channels, out_length - kernel_size + 1),
                output_shape=(out_channels, out_length - kernel_size + 1),
            ),
        )
    )
    _net.add_edges(
        (
            edge(src="input", dst="conv0_i0", src_dst_indices=tuple()),
            edge(src="conv0_i0", dst="conv1_i0", src_dst_indices=tuple()),
            edge(src="conv1_i0", dst="output", src_dst_indices=tuple()),
        )
    )

    return _net
