import json
from os.path import exists, join

import pytest
import torch

import elasticai.creator.nn.fixed_point as nn_creator
from elasticai.creator.file_generation import find_project_root
from elasticai.creator.file_generation.on_disk_path import OnDiskPath
from elasticai.creator.testing.cocotb_runner import run_cocotb_sim


@pytest.mark.simulation
@pytest.mark.slow
@pytest.mark.parametrize("total_bits", [(4, 6, 8, 10)])
def test_build_test_relu_test(total_bits: int) -> None:
    file_name = f"TestRelu_{total_bits}"
    dut = nn_creator.ReLU(total_bits=total_bits)
    val_input = torch.linspace(
        start=-(2 ** (total_bits - 1)),
        end=2 ** (total_bits - 1) - 1,
        steps=2**total_bits,
    )
    val_output = dut(val_input)

    testpattern = {"in": val_input.tolist(), "out": val_output.tolist()}
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
