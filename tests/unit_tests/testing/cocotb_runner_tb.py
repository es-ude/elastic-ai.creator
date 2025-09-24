import cocotb
from cocotb.triggers import Timer


def model(a: int, scale: int, offset: int, do_offset: bool) -> int:
    return scale * a + (0 if not do_offset else offset)


@cocotb.test()
async def model_test(dut):
    A = 1
    B = dut.OFFSET.value.to_signed() if "OFFSET" in dir(dut) else 0
    dut.A.value = A
    await Timer(2, unit="step")
    assert dut.Q.value == model(A, dut.SCALE.value.to_signed(), B, "OFFSET" in dir(dut))
