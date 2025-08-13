import json
from os import makedirs
from os.path import exists, join

import pytest
import torch

import elasticai.creator.nn.fixed_point as nn_creator
from elasticai.creator.file_generation import find_project_root
from elasticai.creator.file_generation.on_disk_path import OnDiskPath
from elasticai.creator.nn.fixed_point.math_operations import FixedPointConfig
from elasticai.creator.testing.cocotb_runner import run_cocotb_sim


@pytest.mark.simulation
@pytest.mark.slow
@pytest.mark.parametrize(
    "total_bits, frac_bits, num_steps", [(6, 4, 32), (8, 4, 32), (10, 9, 64)]
)
def test_build_test_sigmoid_design(
    total_bits: int, frac_bits: int, num_steps: int
) -> None:
    file_name = f"TestSigmoid_{total_bits}_{frac_bits}_{num_steps}"
    fxp = FixedPointConfig(total_bits=total_bits, frac_bits=frac_bits)
    # Build design
    dut = nn_creator.Sigmoid(
        total_bits=total_bits,
        frac_bits=frac_bits,
        num_steps=num_steps,
        sampling_intervall=(fxp.minimum_as_rational, fxp.maximum_as_rational),
    )

    # Build pattern and write into file
    val_input = torch.linspace(
        start=fxp.minimum_as_rational, end=fxp.maximum_as_rational, steps=2 * num_steps
    )
    val_output = dut(val_input)
    testpattern = {
        "in": fxp.cut_as_integer(val_input).int().tolist(),
        "out": fxp.cut_as_integer(val_output).int().tolist(),
    }
    makedirs(f"{find_project_root()}/build_test", exist_ok=True)
    with open(f"{find_project_root()}/build_test/{file_name}.json", "w") as f0:
        json.dump(testpattern, f0, indent=1)

    # Check if design is available
    output_dir = "build_test"
    destination = OnDiskPath(output_dir, parent=find_project_root())
    dut.create_design(file_name).save_to(destination)
    assert exists(join(find_project_root(), output_dir, f"{file_name}.vhd"))

    # Prepare cocotb runner
    set0 = dict(
        src_files=[join(find_project_root(), output_dir, f"{file_name}.vhd")],
        top_module_name=file_name,
        cocotb_test_module="tests.integration_tests.nn.fixed_point.precomputed_tb",
    )
    run_cocotb_sim(**set0)
