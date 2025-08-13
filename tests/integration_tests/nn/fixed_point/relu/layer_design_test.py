import json
from os import makedirs
from os.path import exists, join

import pytest
import torch

import elasticai.creator.nn.fixed_point as nn_creator
from elasticai.creator.file_generation import find_project_root
from elasticai.creator.file_generation.on_disk_path import OnDiskPath
from elasticai.creator.testing.cocotb_runner import run_cocotb_sim


@pytest.mark.simulation
@pytest.mark.slow
@pytest.mark.parametrize("total_bits, num_steps", [(4, 8), (6, 32), (8, 64), (10, 128)])
def test_build_test_relu_test(total_bits: int, num_steps: int) -> None:
    file_name = f"TestRelu_{total_bits}"

    fxp = nn_creator.FixedPointConfig(total_bits=total_bits, frac_bits=1)
    dut = nn_creator.ReLU(total_bits=total_bits)
    val_input = torch.linspace(
        start=fxp.minimum_as_rational, end=fxp.maximum_as_rational, steps=num_steps
    )
    val_output = dut(val_input)

    testpattern = {"in": val_input.tolist(), "out": val_output.tolist()}
    makedirs(f"{find_project_root()}/build_test", exist_ok=True)
    with open(f"{find_project_root()}/build_test/{file_name}.json", "w") as f:
        json.dump(testpattern, f, indent=1)

    output_dir = "build_test"
    destination = OnDiskPath(output_dir, parent=find_project_root())
    dut.create_design(file_name).save_to(destination)
    assert exists(join(find_project_root(), output_dir, f"{file_name}.vhd"))

    set0 = dict(
        src_files=[join(find_project_root(), output_dir, f"{file_name}.vhd")],
        top_module_name=file_name,
        cocotb_test_module="tests.integration_tests.nn.fixed_point.precomputed_tb",
    )
    run_cocotb_sim(**set0)
