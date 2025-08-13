import json

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer

from elasticai.creator.file_generation import find_project_root


@cocotb.test()
async def precomputed_test(dut):
    with open(f"{find_project_root()}/build_test/{dut._name}.json".lower(), "r") as f:
        data = json.load(f)

    clock_period_ns = 10
    dut.enable.value = 1  # Has no impact
    dut.clock.value = 0  # has no impact
    dut.x.value = 0

    cocotb.start_soon(Clock(dut.clock, period=clock_period_ns, units="ns").start())
    await Timer(4 * clock_period_ns, units="ns")
    await RisingEdge(dut.clock)

    chck = list()
    for sig_in, ref_out in zip(data["in"], data["out"]):
        dut.x.value = sig_in
        for _ in range(2):
            await RisingEdge(dut.clock)

        chck.append(dut.y.value.signed_integer == ref_out)
        if not chck[-1]:
            print(f"x={dut.x.value.signed_integer} -> y_pred={dut.y.value.signed_integer}, y_true={ref_out}")

    accuracy = sum(chck) / len(chck)
    print(f"Accuracy of {accuracy*100:.2f}%")
    assert accuracy >= 0.98
