from typing import Callable

import pytest
import torch

from elasticai.creator.arithmetic import FxpConverter, FxpParams
from elasticai.creator.nn.fixed_point.conv1d.design import Conv1dDesign
from elasticai.creator.nn.fixed_point.conv1d.testbench import Conv1dTestbench
from elasticai.creator.vhdl.auto_wire_protocols.port_definitions import create_port
from elasticai.creator.vhdl.design.ports import Port


class DummyConv1d:
    def __init__(self, fxp_params: FxpParams, in_channels: int, out_channels: int):
        self.name: str = "conv1d"
        self.kernel_size: int = 1
        self.input_signal_length = 1
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.port: Port = create_port(
            y_width=fxp_params.total_bits,
            x_width=fxp_params.total_bits,
            x_count=1,
            y_count=2,
        )


def parameters_for_reported_content_parsing(fxp_params, input_expected_pairs):
    def add_expected_prefix_to_pairs(pairs):
        _converter_for_batch = FxpConverter(
            FxpParams(total_bits=8, frac_bits=0, signed=True)
        )  # max for 255 lines of inputs
        pairs_with_prefix = list()
        for i, (pairs_text, pairs_number) in enumerate(pairs):
            pairs_with_prefix.append(list())
            pairs_with_prefix[i].append(list())
            pairs_with_prefix[i].append(pairs_number)
            for batch_number, batch_channel_text in enumerate(pairs_text):
                for out_channel_text in batch_channel_text:
                    for value_text in out_channel_text:
                        pairs_with_prefix[i][0].append(
                            f"result: {_converter_for_batch.integer_to_binary_string_vhdl(batch_number)},"
                            f" {value_text}"
                        )
        return pairs_with_prefix

    pairs_with_prefix = [
        (fxp_params, a, b)
        for a, b in add_expected_prefix_to_pairs(input_expected_pairs)
    ]
    return pairs_with_prefix


@pytest.fixture
def create_uut() -> Callable[[FxpParams, int, int], Conv1dDesign]:
    def create(
        fxp_params: FxpParams, in_channels: int, out_channels: int
    ) -> Conv1dDesign:
        return DummyConv1d(
            fxp_params, in_channels=in_channels, out_channels=out_channels
        )

    return create


@pytest.mark.parametrize(
    "fxp_params, reported, expected",
    (
        parameters_for_reported_content_parsing(
            fxp_params=FxpParams(total_bits=3, frac_bits=0),
            input_expected_pairs=[
                ([[["010"]]], [[[2.0]]]),
                ([[["111"]]], [[[-1.0]]]),
                ([[["001", "010"]]], [[[1.0, 2.0]]]),
                ([[["111", "001"]]], [[[-1.0, 1.0]]]),
            ],
        )
        + parameters_for_reported_content_parsing(
            fxp_params=FxpParams(total_bits=4, frac_bits=1),
            input_expected_pairs=[
                ([[["0001", "1111"]]], [[[0.5, -0.5]]]),
                (
                    [[["0001", "0011"]], [["1000", "1111"]]],
                    [[[0.5, 1.5]], [[-4.0, -0.5]]],
                ),
            ],
        )
    ),
)
def test_parse_reported_content_one_out_channel(
    fxp_params, reported, expected, create_uut
):
    in_channels = None
    out_channels = 1
    bench = Conv1dTestbench(
        name="conv1d_testbench",
        fxp_params=fxp_params,
        uut=create_uut(fxp_params, in_channels, out_channels),
    )
    result = bench.parse_reported_content(reported)
    assert result == expected


@pytest.mark.parametrize(
    "fxp_params, reported, expected",
    (
        parameters_for_reported_content_parsing(
            fxp_params=FxpParams(total_bits=3, frac_bits=0),
            input_expected_pairs=[
                ([[["010"], ["010"]]], [[[2.0], [2.0]]]),
                ([[["001", "010"], ["001", "010"]]], [[[1.0, 2.0], [1.0, 2.0]]]),
                ([[["111", "001"], ["111", "001"]]], [[[-1.0, 1.0], [-1.0, 1.0]]]),
            ],
        )
        + parameters_for_reported_content_parsing(
            fxp_params=FxpParams(total_bits=4, frac_bits=1),
            input_expected_pairs=[
                ([[["0001", "1111"], ["0001", "1111"]]], [[[0.5, -0.5], [0.5, -0.5]]]),
                (
                    [
                        [["0001", "0011"], ["0001", "0011"]],
                        [["1000", "1111"], ["1000", "1111"]],
                    ],
                    [[[0.5, 1.5], [0.5, 1.5]], [[-4.0, -0.5], [-4.0, -0.5]]],
                ),
            ],
        )
    ),
)
def test_parse_reported_content_two_out_channel(
    fxp_params, reported, expected, create_uut
):
    in_channels = None
    out_channels = 2
    bench = Conv1dTestbench(
        name="conv1d_testbench",
        fxp_params=fxp_params,
        uut=create_uut(fxp_params, in_channels, out_channels),
    )
    result = bench.parse_reported_content(reported)
    assert result == expected


def test_input_preparation_with_one_in_channel(create_uut):
    fxp_params = FxpParams(total_bits=3, frac_bits=0)
    in_channels = 1
    out_channels = None
    bench = Conv1dTestbench(
        name="bench_name",
        fxp_params=fxp_params,
        uut=create_uut(fxp_params, in_channels, out_channels),
    )
    input = torch.Tensor([[[1.0, 1.0]]])
    expected = [{"x_0_0": "001", "x_0_1": "001"}]
    result = bench.prepare_inputs(input.tolist())
    assert result == expected


def test_input_preparation_with_two_in_channel(create_uut):
    fxp_params = FxpParams(total_bits=3, frac_bits=0)
    in_channels = 1
    out_channels = None
    bench = Conv1dTestbench(
        name="bench_name",
        fxp_params=fxp_params,
        uut=create_uut(fxp_params, in_channels, out_channels),
    )
    input = torch.Tensor([[[1.0, 1.0], [1.0, 2.0]]])
    expected = [{"x_0_0": "001", "x_0_1": "001", "x_1_0": "001", "x_1_1": "010"}]
    result = bench.prepare_inputs(input.tolist())
    assert result == expected
