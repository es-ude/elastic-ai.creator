import json
from os import makedirs
from os.path import exists, join

import pytest
import torch

import elasticai.creator.nn.fixed_point as nn_creator
from elasticai.creator.file_generation import find_project_root
from elasticai.creator.file_generation.on_disk_path import OnDiskPath
from elasticai.creator.testing.cocotb_runner import run_cocotb_sim


@pytest.mark.slow
@pytest.mark.simulation
@pytest.mark.parametrize("total_bits, frac_bits", [(6, 2)])
def test_build_design_hardsigmoid(total_bits: int, frac_bits: int) -> None:
    file_name = f"TestHardSigmoid_{total_bits}_{frac_bits}"
    dut = nn_creator.HardSigmoid(total_bits=total_bits, frac_bits=frac_bits)
    val_input = torch.linspace(
        start=-(2 ** (total_bits - 1)),
        end=2 ** (total_bits - 1) - 1,
        steps=2**total_bits,
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
