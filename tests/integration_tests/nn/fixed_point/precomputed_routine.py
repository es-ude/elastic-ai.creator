import json
from os.path import exists

import torch

from elasticai.creator.file_generation import find_project_root
from elasticai.creator.file_generation.on_disk_path import OnDiskPath
from elasticai.creator.nn.fixed_point.math_operations import FixedPointConfig
from elasticai.creator.testing.cocotb_runner import (
    get_and_create_sim_build_dir,
    run_cocotb_sim,
)


def routine_testing_precomputed_module(
    dut, num_steps: int, fxp: FixedPointConfig, file_name: str, file_suffix: str = "vhd"
) -> None:
    # Build pattern and write into file
    val_input = fxp.as_rational(
        fxp.round_to_integer(
            torch.linspace(
                start=fxp.minimum_as_rational,
                end=fxp.maximum_as_rational,
                steps=2 * num_steps,
            )
        )
    )
    val_output = dut(val_input)
    testpattern = {
        "in": fxp.cut_as_integer(val_input).int().tolist(),
        "out": fxp.round_to_integer(val_output).int().tolist(),
    }

    output_dir = f"build_test/{file_name.lower()}"
    build_dir = get_and_create_sim_build_dir(output_dir)
    with open(build_dir / f"{file_name}.json".lower(), "w") as f0:
        json.dump(testpattern, f0, indent=1)

    # Check if design is available
    destination = OnDiskPath(output_dir, parent=find_project_root())
    dut.create_design(file_name).save_to(destination)
    assert exists(build_dir / f"{file_name}.{file_suffix}")
    assert exists(build_dir / f"{file_name}.json".lower())

    # Prepare cocotb runner
    run_cocotb_sim(
        src_files=[build_dir / f"{file_name}.{file_suffix}"],
        top_module_name=file_name,
        cocotb_test_module="tests.integration_tests.nn.fixed_point.precomputed_tb",
        build_waveforms=True,
    )
