from pathlib import Path

import cocotb
from cocotb.triggers import Timer

from elasticai.creator.file_generation.resource_utils import get_full_path
from elasticai.creator.testing.cocotb_runner import run_cocotb_sim_for_src_dir

_path2src = Path(
    get_full_path("tests.unit_tests.testing", "cocotb_runner_tb.py")
).parent
_path2tb = "tests.unit_tests.testing.cocotb_runner_tb"

cocotb_settings_verilog0 = dict(
    src_files=["cocotb_runner_tb.v"],
    top_module_name="COCOTB_TEST",
    cocotb_test_module=_path2tb,
    path2src=_path2src,
    defines={},
    params={"BITWIDTH": 4, "SCALE": 2},
    en_debug_mode=True,
)
cocotb_settings_verilog1 = dict(
    src_files=["cocotb_runner_tb.v"],
    top_module_name="COCOTB_TEST",
    cocotb_test_module=_path2tb,
    path2src=_path2src,
    params={"BITWIDTH": 4, "SCALE": 2, "OFFSET": 2},
    defines={"DEFINE_TEST": True},
    en_debug_mode=True,
)
cocotb_settings_vhdl0 = dict(
    src_files=["cocotb_runner_tb.vhd"],
    top_module_name="COCOTB_TEST",
    cocotb_test_module=_path2tb,
    path2src=_path2src,
    params={"BITWIDTH": 4, "SCALE": 2},
    defines={},
    en_debug_mode=True,
)
cocotb_settings_vhdl1 = dict(
    src_files=["cocotb_runner_tb.vhd"],
    top_module_name="COCOTB_TEST",
    cocotb_test_module=_path2tb,
    path2src=_path2src,
    params={"BITWIDTH": 4, "SCALE": 2, "OFFSET": 2},
    defines={"DEFINE_TEST": True},
    en_debug_mode=True,
)


def model(a: int, scale: int, offset: int, do_offset: bool) -> int:
    return scale * a + (0 if not do_offset else offset)


@cocotb.test()
async def model_test(dut):
    print(dir(dut))
    A = 1
    B = dut.OFFSET.value if "OFFSET" in dir(dut) else 0
    dut.A.value = A
    await Timer(2, units="step")
    assert dut.Q.value == model(A, dut.SCALE.value, B, "OFFSET" in dir(dut))


if __name__ == "__main__":
    run_cocotb_sim_for_src_dir(**cocotb_settings_verilog0)
    run_cocotb_sim_for_src_dir(**cocotb_settings_verilog1)
    run_cocotb_sim_for_src_dir(**cocotb_settings_vhdl0)
    run_cocotb_sim_for_src_dir(**cocotb_settings_vhdl1)
