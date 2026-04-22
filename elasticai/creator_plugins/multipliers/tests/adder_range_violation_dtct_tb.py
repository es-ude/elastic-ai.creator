from pathlib import Path

import cocotb
import elasticai.creator_plugins.mult as test_dut
from cocotb.triggers import Timer

from elasticai.creator.testing.cocotb_runner import run_cocotb_sim_for_src_dir

cocotb_settings = dict(
    src_files=["adder_range_violation_dtct.v"],
    path2src=Path(test_dut.__file__).parent / "verilog",
    top_module_name="ADDER_RANGE_DTCT",
    cocotb_test_module="elasticai.creator_plugins.mult.tests.adder_range_violation_dtct_tb",
    params={"BITWIDTH": 9, "NUM_ADDERS": 0, "SIGNED": 1},
)


def dtct_model(
    a: int, bitwidth: int, num_adders: int, signed: bool
) -> tuple[int, int, int]:
    valrange = (
        [0, 2 ** (bitwidth - num_adders) - 1]
        if not signed
        else [-(2 ** (bitwidth - 1 - num_adders)), 2 ** (bitwidth - 1 - num_adders) - 1]
    )
    down = a < valrange[0]
    upper = a > valrange[1]
    return 1 if down else 0, 1 if upper else 0, 0 if down or upper else 1


@cocotb.test()
async def violation_chck(dut):
    valrange = (
        [0, 2 ** (dut.BITWIDTH.value.to_unsigned()) - 1]
        if not dut.SIGNED.value
        else [
            -(2 ** (dut.BITWIDTH.value.to_unsigned() - 1)),
            2 ** (dut.BITWIDTH.value.to_unsigned() - 1) - 1,
        ]
    )
    for idx in range(valrange[0], valrange[1]):
        dut.A.value = idx

        await Timer(2, unit="step")
        assert (
            dut.DOWNER_LIMIT.value,
            dut.UPPER_LIMIT.value,
            dut.DATA_VALID.value,
        ) == dtct_model(
            idx,
            dut.BITWIDTH.value.to_unsigned(),
            dut.NUM_ADDERS.value.to_unsigned(),
            dut.SIGNED.value,
        )


if __name__ == "__main__":
    run_cocotb_sim_for_src_dir(**cocotb_settings)
