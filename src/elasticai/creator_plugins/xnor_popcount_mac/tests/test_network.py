from typing import Literal

import cocotb
import hypothesis
import pytest
import torch
import torch.nn as nn
from cocotb.clock import Clock
from cocotb.types import LogicArray
from hypothesis import strategies
from hypothesis.strategies import DataObject

from elasticai.creator import ir
from elasticai.creator.testing.cocotb_pytest import CocotbTestFixture, eai_testbench
from elasticai.creator.testing.cocotb_stream import ResetControl, StreamInterface
from elasticai.creator_plugins.xnor_popcount_mac.src.nn import conv1d
from elasticai.creator_plugins.xnor_popcount_mac.tests._common import (
    CNNBuilder,
    build_design,
    collect_all_srcs_from_build_dir,
)


@cocotb.test()
@eai_testbench
async def check_network_with_bin_filter(dut, num_layers, input_chunks, expected_chunks):
    cocotb.start_soon(Clock(dut.clk, period=10).start())
    reset = ResetControl.from_dut(dut)
    await reset.reset_active_high(reset_cycles=2)
    dut.en.value = 1
    stream = StreamInterface.from_dut(
        dut,
        input_to_logic_array=lambda x: LogicArray(x),
        output_from_value=output_from_value,
    )
    collection = cocotb.start_soon(
        stream.collect_chunks(expected_count=len(expected_chunks), max_cycles=100)
    )
    await stream.drive_chunks([bytes.fromhex(c) for c in input_chunks])
    observed = await collection
    assert [bytes.fromhex(c) for c in expected_chunks] == observed


@hypothesis.given(data_generator=strategies.data())
@hypothesis.settings(
    suppress_health_check=[hypothesis.HealthCheck.function_scoped_fixture],
    max_examples=5,
    deadline=5000,
)
@pytest.mark.simulation
@pytest.mark.parametrize("num_layers", [1, 2, 3])
def test_network(
    cocotb_test_fixture: CocotbTestFixture,
    data_generator: strategies.DataObject,
    num_layers: int,
):
    build_dir = cocotb_test_fixture.get_artifact_dir()
    kernel_size = 2
    data_out_depth = 2
    sw_model = nn.Sequential(*[conv1d(kernel_size, 1, 1) for _ in range(num_layers)])

    registry = _to_ir(sw_model, data_out_depth)
    network_graph = registry["network"]

    skeleton = registry["skeleton"]
    data_in_depth = int(skeleton.attributes["generic_map"]["DATA_IN_DEPTH"])

    input_bits = _sample_input_bits(data_generator, num_samples=data_in_depth)
    input_tensor = bit_vector_to_tensor(input_bits)

    with torch.no_grad():
        output_tensor = sw_model(input_tensor)

    output_bits = tensor_to_bit_vector(output_tensor)
    vhdl_dir = build_dir / "vhdl"
    vhdl_dir.mkdir(exist_ok=True)
    build_design(network_graph, registry, vhdl_dir)
    cocotb_test_fixture.write(
        {
            "input_chunks": [bytes([b]).hex() for b in input_bits],
            "expected_chunks": [bytes([b]).hex() for b in output_bits],
        }
    )

    cocotb_test_fixture.set_srcs(collect_all_srcs_from_build_dir(vhdl_dir))
    cocotb_test_fixture.set_top_module_name("network")
    cocotb_test_fixture.run(params={}, defines={})


def _sample_input_bits(data: DataObject, num_samples: int):
    return data.draw(
        strategies.lists(
            strategies.integers(0, 1), min_size=num_samples, max_size=num_samples
        )
    )


def _to_ir(sw_model: nn.Sequential, data_out_depth: int) -> ir.Registry:
    builder = CNNBuilder(data_out_depth=data_out_depth)

    for weight_str in _get_weights_for_hw(sw_model):
        builder.add_conv(weight_str)
    return builder.build()


def _get_weights_for_hw(sw_model: nn.Sequential) -> list[str]:
    weight_strings = []
    with torch.no_grad():
        for layer in sw_model:
            weight_tensor = layer.weight.flatten()  # type: ignore
            weight_str = "".join(["1" if w >= 0 else "0" for w in weight_tensor])
            weight_strings.append(weight_str)
    return weight_strings


def bit_vector_to_tensor(logic: list[Literal[0, 1] | int]) -> torch.Tensor:
    return (2 * torch.tensor(logic) - 1).view(1, 1, len(logic)).to(torch.float)


def tensor_to_bit_vector(input: torch.Tensor) -> list[int]:
    return (input >= 0).to(torch.uint8).flatten().tolist()


def output_from_value(x: LogicArray) -> bytes:
    return x.to_bytes(byteorder="big")


def input_to_logic_array(x: bytes) -> LogicArray:
    return LogicArray.from_bytes(x, byteorder="big")
