import dataclasses
import logging
from dataclasses import dataclass
from pathlib import Path
from pprint import pformat

import cocotb
import elasticai.creator_plugins.lutron_filter as lf
import pytest
import torch
from cocotb import clock as cocotbclk
from cocotb.triggers import RisingEdge
from elasticai.experiment_framework import remote_control as rc
from elasticai.experiment_framework import synthesis as eaixp_synth
from torch import nn as tnn

from elasticai.creator.ir import IrSerializer
from elasticai.creator.ir2vhdl import Ir2Vhdl, PluginLoader, Shape
from elasticai.creator.testing import (
    CocotbTestFixture,
    HWTester,
    ResetControl,
    StreamInterface,
    eai_testbench,
)
from elasticai.creator_plugins.grouped_filter import FilterParameters
from elasticai.creator_plugins.skeleton.hw_function_id import HwFunctionIdUpdater


def pretty_log_debug(g):
    serialized = IrSerializer().serialize(g)
    logger.debug(pformat(serialized), stacklevel=2)


def specify_split(
    p: lf.FilterParameters,
) -> tuple[lf.FilterParameters, lf.FilterParameters]:
    return FilterParameters(
        kernel_size=p.kernel_size,
        in_channels=p.in_channels,
        out_channels=p.in_channels,
        groups=p.in_channels // 2 if p.in_channels % 2 == 0 else p.in_channels,
    ), FilterParameters(
        kernel_size=1,
        in_channels=p.in_channels,
        out_channels=p.out_channels,
        groups=1,
    )


logger = logging.getLogger()


@cocotb.test
@eai_testbench
async def check_network(
    dut, input: tuple[int, ...], expected: tuple[int, ...], **kwargs
) -> None:
    cocotb.start_soon(cocotbclk.Clock(dut.clk, 10).start())
    stream = StreamInterface.from_dut(dut)
    reset = ResetControl.from_dut(dut)
    dut.src_valid.value = 0
    dut.dst_ready.value = 0

    await RisingEdge(dut.clk)
    await reset.reset_active_high()
    dut.en.value = 1
    collect_task = cocotb.start_soon(
        stream.collect_chunks(
            expected_count=len(expected), max_cycles=4 * len(expected)
        )
    )
    await stream.drive_chunks(
        [
            f"{int.from_bytes(n.to_bytes(signed=True)[0:3], signed=False):03b}"[-3:]
            for n in input
        ]
    )
    observed = await collect_task
    expected_bits = "".join([f"{n:1b}" for n in expected])
    assert "".join(observed) == expected_bits


@dataclass
class InputOutputData:
    expected: tuple[int, ...]
    input: tuple[int, ...]
    expected_per_layer: dict[str, tuple[int, ...]]


def setup(
    seed: int, tmpdir: Path, convert_to_vhdl: Ir2Vhdl | None = None
) -> InputOutputData:
    channels = [4, 8]
    model = tnn.Sequential(
        tnn.Conv1d(1, out_channels=channels[0], kernel_size=1),
        tnn.PReLU(),
        tnn.Conv1d(channels[0], channels[1], kernel_size=3),
        tnn.PReLU(),
        tnn.Conv1d(channels[1], 2, kernel_size=1),
        tnn.Flatten(),
        tnn.PReLU(),
        tnn.Linear(2, 2),
        tnn.PReLU(),
    )
    torch.manual_seed(seed)
    metadata = {
        "required_input_size": lf.compute_required_input_size(model, 1),
        "num_input_bits": 3,
        "in_channels": 1,
    }
    tmpdir.mkdir(exist_ok=True)
    prepared_model = lf.prepare_for_training(
        model, specify_split, tmpdir / "prepared_for_training"
    )
    lf.translate(
        prepared_model,
        Shape(metadata["in_channels"], metadata["required_input_size"]),
        metadata["num_input_bits"],
        tmpdir,
        convert_to_vhdl,
    )
    num_bits = 3
    input_tensor = torch.tensor(
        [
            [
                [
                    -(2 ** (num_bits - 1)) + n
                    for n in range(metadata["required_input_size"])
                ]
            ]
        ],
        dtype=torch.float32,
    )

    expected_per_layer = {}
    index = 0

    def make_recorder(name):
        def record(module, arg, output):
            nonlocal index
            tmp = output.detach()
            tmp = ((1 + tmp) / 2).to(torch.int)
            if len(tmp.shape) > 2:
                tmp = tmp.transpose(1, 2)
            expected_per_layer[index] = tuple(tmp.tolist())
            index += 1
            return output

        return record

    prepared_model.eval()

    for name, m in prepared_model.named_children():
        if type(m).__name__ == "Binarize":
            m.register_forward_hook(make_recorder(name))
    expected_tensor = prepared_model(input_tensor)
    expected_tensor = (1 + expected_tensor) / 2

    def to_python(x: list | float | int) -> tuple | int:
        if isinstance(x, list):
            return tuple(to_python(y) for y in x)
        return int(x)

    if len(expected_tensor.shape) > 2:
        expected_tensor = expected_tensor.transpose(1, 2)

    return InputOutputData(
        input=tuple(to_python(input_tensor.tolist())[0][0]),  # type: ignore[index]  # ty:ignore[not-subscriptable]
        expected=tuple(to_python(expected_tensor.flatten().tolist())),  # type: ignore
        expected_per_layer=expected_per_layer,  # type: ignore
    )


@pytest.mark.slow
@pytest.mark.simulation
@pytest.mark.parametrize(
    "seed",
    [
        3,
    ],
)
def test_toy_model_simulates_correctly(
    cocotb_test_fixture: CocotbTestFixture, seed: int
):

    test_data = setup(seed, cocotb_test_fixture.get_artifact_dir())
    cocotb_test_fixture.write(dataclasses.asdict(test_data))
    cocotb_test_fixture.set_srcs_from_artifact_dir("vhdl/*.vhd")
    cocotb_test_fixture.set_top_module_name("network")
    cocotb_test_fixture.run(params={}, defines={})


def _synthesize(src_dir: Path) -> Path:
    _synth = eaixp_synth.CachedVivadoSynthesis()
    return _synth.synthesize(src_dir) / "results/impl/env5_top_reconfig.bin"


@pytest.mark.hardware
def test_toy_model_on_hw(tmp_path):
    seed = 3
    convert_to_vhdl = Ir2Vhdl()
    loader = PluginLoader(convert_to_vhdl)
    loader.load_from_package("middleware")
    build_dir = tmp_path / "vhdl"
    test_data = setup(seed, tmp_path, convert_to_vhdl)
    hwid_updater = HwFunctionIdUpdater(build_dir)
    hwid_updater.compute_id()
    hwid_updater.write_id()
    device = rc.probe_for_devices()[0]
    tester = HWTester(
        synth_fn=_synthesize, device=rc.remote_control.connect_remote_control(device)
    )
    with tester.prepare_hw_function(build_dir, hwid_updater.id) as run_inference:
        inputs = bytearray()
        for n in test_data.input:
            inputs.extend(n.to_bytes(1, signed=True))
        predictions = run_inference(bytes(inputs), len(test_data.expected))
    expected_words = bytearray()
    for n in test_data.expected:
        expected_words.extend(n.to_bytes(1, signed=False))
    assert predictions == bytes(expected_words)
