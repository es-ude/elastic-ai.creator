from os.path import exists
from pathlib import Path

import torch

from elasticai.creator.arithmetic import (
    FxpArithmetic,
)
from elasticai.creator.file_generation import find_project_root
from elasticai.creator.file_generation.on_disk_path import OnDiskPath
from elasticai.creator.testing import (
    build_report_folder_and_testdata,
    run_cocotb_sim,
)


def routine_testing_precomputed_module(
    dut,
    num_steps: int,
    fxp: FxpArithmetic,
    file_name: str,
    file_suffix: str = "vhd",
) -> None:
    val_input = fxp.as_rational(
        fxp.round_to_integer(
            torch.linspace(
                start=fxp.minimum_as_rational,
                end=fxp.maximum_as_rational,
                steps=2 * num_steps,
            )
        )
    )
    dut.eval()
    with torch.no_grad():
        val_output = dut(val_input)

    output_dir = build_report_folder_and_testdata(
        dut_name=file_name,
        testdata={
            "in": fxp.cut_as_integer(val_input).int().tolist(),
            "out": fxp.round_to_integer(val_output).int().tolist(),
        },
    )

    output_dir_sub = Path(*output_dir.parts[len(find_project_root().parts) :])
    destination = OnDiskPath(str(output_dir_sub), parent=find_project_root())
    dut.create_design(file_name).save_to(destination)
    assert exists(output_dir / f"{file_name}.{file_suffix}")
    assert exists(output_dir / "testdata.json")

    run_cocotb_sim(
        src_files=[output_dir / f"{file_name}.{file_suffix}"],
        top_module_name=file_name,
        cocotb_test_module="tests.integration_tests.nn.fixed_point.precomputed_tb",
        waveform_save_dst=str(output_dir),
    )
