from os.path import exists
from pathlib import Path

import pytest
import torch

from elasticai.creator.arithmetic import FxpArithmetic, FxpParams
from elasticai.creator.file_generation import find_project_root
from elasticai.creator.file_generation.on_disk_path import OnDiskPath
from elasticai.creator.testing import (
    build_report_folder_and_testdata,
    run_cocotb_sim,
)
from elasticai.creator.vhdl.shared_designs.rom.design import Rom


@pytest.mark.simulation
@pytest.mark.parametrize(
    "total_bits, frac_bits, features_in", [(4, 2, 8), (8, 2, 12), (10, 4, 63)]
)
def test_build_test_rom(
    total_bits: int,
    frac_bits: int,
    features_in: int,
) -> None:
    file_name = f"TestROM_{total_bits}_{frac_bits}_{features_in}"

    # Build pattern and write into file
    fxp = FxpArithmetic(FxpParams(total_bits=total_bits, frac_bits=frac_bits))
    val_input = (
        fxp.as_rational(
            torch.randint(
                low=fxp.config.minimum_as_integer,
                high=fxp.config.maximum_as_integer + 1,
                size=(features_in,),
            )
        )
        .int()
        .tolist()
    )

    output_dir = build_report_folder_and_testdata(
        dut_name=file_name,
        testdata={
            "data": val_input,
        },
    )

    # Build design and check if design is available
    output_dir_sub = Path(*output_dir.parts[len(find_project_root().parts) :])
    destination = OnDiskPath(str(output_dir_sub), parent=find_project_root())
    Rom(name=file_name, data_width=total_bits, values_as_integers=val_input).save_to(
        destination
    )
    assert exists(output_dir / f"{file_name}.vhd")
    assert exists(output_dir / "testdata.json")

    # Prepare cocotb runner and save waveforms
    run_cocotb_sim(
        src_files=[output_dir / f"{file_name}.vhd"],
        top_module_name=file_name,
        cocotb_test_module="tests.integration_tests.share_designs.rom.rom_tb",
        waveform_save_dst=str(output_dir),
    )
