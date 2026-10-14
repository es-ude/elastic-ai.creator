from pathlib import Path

import pytest
import torch

from elasticai.creator.file_generation.on_disk_path import OnDiskPath
from elasticai.creator.nn.integer.depthconv1d import DepthConv1d
from elasticai.creator.testing.ghdl_simulation import GHDLSimulator, SimulationError

pytestmark = pytest.mark.simulation

NAME = "depthconv1d_0"


def simulate(build_dir: Path) -> None:
    simulator = GHDLSimulator(workdir=str(build_dir), top_design_name=f"{NAME}_tb")
    simulator.initialize()
    # The testbench stops the simulation with an assertion failure once all
    # inputs are processed, so GHDL always exits with an error.
    with pytest.raises(SimulationError, match="Simulation done"):
        simulator.run()


def read_integers(path: Path) -> list[int]:
    return [int(line) for line in path.read_text().split()]


@pytest.mark.parametrize(
    "padding",
    [
        pytest.param(0, id="no_padding"),
        # BUG: the templates do not saturate the outputs. One output is 128
        # before int_forward clamps it to 127, and the hardware wraps it to 0.
        pytest.param(1, id="padding_1", marks=pytest.mark.xfail(strict=True)),
        pytest.param("same", id="padding_same", marks=pytest.mark.xfail(strict=True)),
    ],
)
def test_hardware_matches_int_forward(tmp_path: Path, padding: int | str) -> None:
    build_dir = tmp_path / "build"
    data_dir = build_dir / "data"
    data_dir.mkdir(parents=True)
    torch.manual_seed(0)
    layer = DepthConv1d(
        name=NAME,
        in_channels=3,
        out_channels=3,
        kernel_size=3,
        padding=padding,
        seq_len=8,
        quant_bits=8,
        quant_data_dir=str(data_dir),
    )
    # Two samples, so that the testbench runs the layer twice and the state
    # must be reset in between.
    inputs = torch.randn(2, 3, 8)
    layer.train()
    for _ in range(3):
        layer(inputs)
    layer.eval()
    layer.precompute()
    # Writes the inputs and expected outputs that the testbench reads.
    layer.int_forward(layer.inputs_QParams.quantize(inputs))
    layer.create_design(NAME).save_to(OnDiskPath("build", parent=str(tmp_path)))

    simulate(build_dir)

    expected = read_integers(data_dir / f"{NAME}_q_y.txt")
    actual = read_integers(data_dir / f"{NAME}_q_out.txt")
    assert actual == expected
