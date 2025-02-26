import pytest

from elasticai.creator.ir2vhdl import LoweringPass, Shape, edge, vhdl_node
from elasticai.creator_plugins.time_multiplexed_sequential.src import (
    network as _network_handler,
)
from elasticai.creator_plugins.time_multiplexed_sequential.src import (
    sequential,
)

from .high_level_network import network


@pytest.fixture
def lower():
    _lower = LoweringPass()
    _lower.register(sequential)
    _lower.register_iterable(_network_handler)
    return _lower


def test_sets_input_correctly_for_version_without_sliding_window(lower):
    lowered = tuple(
        lower(
            [
                network(
                    input_shape=(1, 3),
                    kernel_size=3,
                    out_channels=1,
                ),
            ]
        )
    )[0]
    expected_input_node = vhdl_node(
        name="input",
        type="input",
        implementation="",
        input_shape=(1, 3),
        output_shape=(1, 3),
    )
    assert lowered.nodes["input"] == expected_input_node


def test_first_shift_reg_has_three_output_wires():
    in_channels = 2
    length = 4
    _net = network(
        input_shape=(in_channels, length), kernel_size=2, out_channels=2, stride=1
    )

    lowered = sequential(_net)
    node = lowered.nodes["shift_register_i0"]
    assert node.output_shape == (2, 2)


def test_shift_reg_predecessor_is_conv1():
    in_channels = 2
    length = 4
    _net = network(
        input_shape=(in_channels, length), kernel_size=2, out_channels=2, stride=1
    )

    lowered = sequential(_net)
    p = {n.name for n in (iter(lowered.successors("shift_register_i0").values()))}

    assert p == {"conv1_i0"}


def test_second_conv_has_three_input_wires():
    kernel_size = 1
    _net = network(
        input_shape=(3, 2), kernel_size=kernel_size, out_channels=3, stride=1
    )
    hl_conv1 = _net.nodes["conv1_i0"]
    lowered = sequential(_net)
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
        lowered = sequential(_net)
        low_level_conv1 = lowered.nodes["conv1_i0"]
        assert low_level_conv1.input_shape.depth == hl_conv1.input_shape.depth


def test_add_correct_edges(lower):
    net = network(
        input_shape=(2, 7),
        kernel_size=3,
        out_channels=1,
    )
    lowered = tuple(lower([net]))[0]
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


def test_keeps_input_shape_for_first_conv(lower):
    lowered = tuple(
        lower(
            [
                network(
                    input_shape=(1, 4),
                    kernel_size=3,
                    out_channels=1,
                ),
            ]
        )
    )[0]
    input_shape = lowered.nodes["conv0_i0"].input_shape
    assert input_shape == lowered.nodes["input"].output_shape


def test_generates_adapter_for_addressing_skeleton(lower):
    lowered = {
        i.name: i
        for i in lower(
            [
                network(
                    input_shape=(1, 4),
                    kernel_size=4,
                    out_channels=1,
                ),
            ]
        )
    }

    adapter = lowered["buffered_network_wrapper"]
    assert adapter.attributes["generic_map"] == {
        "KERNEL_SIZE": "4",
        "STRIDE": "2",
    }


def test_generates_adapter_for_addressing_skeleton_with_varied_parameters(lower):
    in_channels = 2
    width = 10
    kernel_size = 3
    expected_adapter_kernel_size = kernel_size * in_channels
    expected_adapter_stride = in_channels
    lowered = {
        i.name: i
        for i in lower(
            [
                network(
                    input_shape=(in_channels, width),
                    kernel_size=kernel_size,
                    out_channels=4,
                    stride=1,
                ),
            ]
        )
    }

    adapter = lowered["buffered_network_wrapper"]
    assert adapter.attributes["generic_map"] == {
        "KERNEL_SIZE": str(expected_adapter_kernel_size),
        "STRIDE": str(expected_adapter_stride),
    }
