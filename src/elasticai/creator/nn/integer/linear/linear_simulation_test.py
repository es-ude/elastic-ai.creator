from pathlib import Path
from typing import Any

import pytest
import torch

from elasticai.creator.file_generation.on_disk_path import OnDiskPath
from elasticai.creator.nn.integer.linear import Linear
from elasticai.creator.testing.ghdl_simulation import GHDLSimulator, SimulationError

pytestmark = pytest.mark.simulation

NAME = "linear_0"


def train_and_precompute(layer: Linear, inputs: torch.Tensor) -> None:
    layer.train()
    for _ in range(3):
        layer(inputs)
    layer.eval()
    layer.precompute()


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
    "in_features, out_features, options",
    [
        pytest.param(4, 3, dict(), id="1d"),
        pytest.param(4, 3, dict(num_dimensions=2), id="2d"),
        pytest.param(
            1,
            3,
            dict(num_dimensions=2),
            id="2d_one_input_feature",
        ),
        pytest.param(
            4,
            4,
            dict(num_dimensions=3, use_parallelised_template=True, unroll_factor=4),
            id="2d_parallel",
        ),
        pytest.param(
            4,
            4,
            dict(num_dimensions=2, use_parallelised_template=True, unroll_factor=2),
            id="2d_parallel_saturating",
        ),
        pytest.param(
            4,
            4,
            dict(
                num_dimensions=2,
                use_parallelised_template=True,
                unroll_factor=2,
                use_pipeline_template=True,
            ),
            id="2d_pipeline",
        ),
    ],
)
def test_hardware_matches_int_forward(
    tmp_path: Path, in_features: int, out_features: int, options: dict[str, Any]
) -> None:
    build_dir = tmp_path / "build"
    data_dir = build_dir / "data"
    data_dir.mkdir(parents=True)
    torch.manual_seed(0)
    layer = Linear(
        name=NAME,
        in_features=in_features,
        out_features=out_features,
        quant_bits=8,
        quant_data_dir=str(data_dir),
        **options,
    )
    num_dimensions = options.get("num_dimensions")
    shape = (5,) if num_dimensions is None else (5, num_dimensions)
    inputs = torch.randn(*shape, in_features)
    train_and_precompute(layer, inputs)
    # Writes the inputs and expected outputs that the testbench reads.
    layer.int_forward(layer.inputs_QParams.quantize(inputs))
    layer.create_design(NAME).save_to(OnDiskPath("build", parent=str(tmp_path)))

    simulate(build_dir)

    expected = read_integers(data_dir / f"{NAME}_q_y.txt")
    actual = read_integers(data_dir / f"{NAME}_q_out.txt")
    assert actual == expected
