from glob import glob
from itertools import chain
from os.path import exists
from pathlib import Path

import numpy as np
import torch

from elasticai.creator.arithmetic import FxpArithmetic
from elasticai.creator.file_generation import find_project_root
from elasticai.creator.file_generation.on_disk_path import OnDiskPath
from elasticai.creator.nn import Sequential
from elasticai.creator.testing import (
    build_report_folder_and_testdata,
    run_cocotb_sim,
)


def routine_testing_sequential_module(
    dut: Sequential,
    feat_in: int,
    fxp: FxpArithmetic,
    file_name: str,
    check_quant: bool = True,
) -> None:
    val_input = fxp.as_rational(
        torch.randint(
            low=-(2 ** (fxp.total_bits - 2)),
            high=2 ** (fxp.total_bits - 2),
            size=(20, feat_in),
        )
    )

    weights_q, bias_q = dut[0].get_params_quant()
    if check_quant:
        weights_f, bias_f = dut[0].get_params()
        error_w = (
            np.array(weights_f)
            - np.array(weights_q) * fxp.config.minimum_step_as_rational
        )
        assert np.all(np.abs(error_w) < fxp.config.minimum_step_as_rational)
        error_b = (
            np.array(bias_f) - np.array(bias_q) * fxp.config.minimum_step_as_rational
        )
        assert np.all(np.abs(error_b) < fxp.config.minimum_step_as_rational)

    dut.eval()
    with torch.no_grad():
        val_output = dut(val_input)
    output_dir = build_report_folder_and_testdata(
        dut_name=file_name,
        testdata={
            "in": fxp.cut_as_integer(val_input).int().tolist(),
            "out": fxp.round_to_integer(val_output).int().tolist(),
            "params": list(
                chain.from_iterable([a + [b] for a, b in zip(weights_q, bias_q)])
            ),
        },
    )

    output_dir_sub = Path(*output_dir.parts[len(find_project_root().parts) :])
    destination = OnDiskPath(str(output_dir_sub), parent=find_project_root())
    dut.create_design(file_name).save_to(destination)
    assert exists(output_dir / f"{file_name}.vhd")
    assert exists(output_dir / "testdata.json")

    # --- Prepare and start cocotb runner
    src_folders = [folder for folder in output_dir.iterdir() if folder.is_dir()]
    src_files = [output_dir / f"{file_name}.vhd"]
    for folder in src_folders:
        src_files.extend(glob(str(output_dir / folder / "*.vhd")))
    run_cocotb_sim(
        src_files=src_files,
        top_module_name=file_name,
        cocotb_test_module="tests.integration_tests.nn.fixed_point.sequential_tb",
        waveform_save_dst=str(output_dir),
    )
