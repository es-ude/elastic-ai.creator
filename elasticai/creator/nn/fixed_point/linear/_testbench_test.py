from typing import Callable

import pytest
import torch

from elasticai.creator.vhdl.auto_wire_protocols.port_definitions import create_port
from elasticai.creator.vhdl.design.ports import Port

from ..number_converter import FXPParams, NumberConverter
from .testbench import LinearTestbench
from .design import LinearDesign


class DummyLinear:
    def __init__(
        self, fxp_params: FXPParams, in_signal_length: int, out_signal_length: int
    ):
        self.name: str = "linear0"
        self.in_feature_num = in_signal_length
        self.out_feature_num = out_signal_length
        self.data_width = fxp_params.total_bits
        self.frac_width = fxp_params.frac_bits
        self.port: Port = create_port(
            y_width=fxp_params.total_bits,
            x_width=fxp_params.total_bits,
            x_count=1,
            y_count=2,
        )


def parameters_for_reported_content_parsing(fxp_params, input_expected_pairs):
    def add_expected_prefix_to_pairs(pairs):
        _converter_for_batch = NumberConverter(
            FXPParams(8, 0)
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
                            f"result: {_converter_for_batch.integer_to_bits(batch_number)},"
                            f" {value_text}"
                        )
        return pairs_with_prefix

    pairs_with_prefix = [
        (fxp_params, a, b)
        for a, b in add_expected_prefix_to_pairs(input_expected_pairs)
    ]
    return pairs_with_prefix


@pytest.fixture
def create_uut() -> Callable[[FXPParams, int, int], LinearDesign]:
    def create(
        fxp_params: FXPParams, in_signal_length: int, out_signal_length: int
    ) -> LinearDesign:
        return DummyLinear(
            fxp_params=fxp_params,
            in_signal_length=in_signal_length,
            out_signal_length=out_signal_length,
        )

    return create


@pytest.mark.parametrize(
    "fxp_params, reported, y",
    (
        parameters_for_reported_content_parsing(
            fxp_params=FXPParams(total_bits=3, frac_bits=0),
            input_expected_pairs=[
                ([[["010"]]], [[[2.0]]]),
                ([[["001", "010"]]], [[[1.0, 2.0]]]),
                ([[["111", "001"]]], [[[-1.0, 1.0]]]),
            ],
        )
        + parameters_for_reported_content_parsing(
            fxp_params=FXPParams(total_bits=4, frac_bits=1),
            input_expected_pairs=[
                ([[["0001", "1111"]]], [[[0.5, -0.5]]]),
                ([[["0001", "0011", "1000", "1111"]]], [[[0.5, 1.5, -4.0, -0.5]]]),
            ],
        )
    ),
)
def test_parse_reported_content_one_out_channel(fxp_params, reported, y, create_uut):
    in_signal_length = None
    out_signal_length = 1
    bench = LinearTestbench(
        name="linear_testbench",
        uut=create_uut(fxp_params, in_signal_length, out_signal_length),
    )
    print(f"{reported=}")
    assert y == bench.parse_reported_content(reported)


def test_input_preparation_with_one_in_channel(create_uut):
    fxp_params = FXPParams(total_bits=3, frac_bits=0)
    in_signal_length = 1
    out_signal_length = None
    bench = LinearTestbench(
        name="linear_testbench",
        uut=create_uut(fxp_params, in_signal_length, out_signal_length),
    )
    input = torch.Tensor([[[1.0, 1.0]]])
    expected = [{"x_0_0": "001", "x_0_1": "001"}]
    assert expected == bench.prepare_inputs(input.tolist())
