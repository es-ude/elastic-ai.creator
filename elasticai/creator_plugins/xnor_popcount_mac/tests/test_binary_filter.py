import cocotb
import pytest
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge

from elasticai.creator import ir2vhdl as ir
from elasticai.creator.ir import Registry, attribute
from elasticai.creator.testing.cocotb_pytest import (
    CocotbTestFixture,
    eai_testbench,
)
from elasticai.creator.testing.cocotb_stream import (
    ClockReset,
    StreamInterface,
)

factory = ir.IrFactory()

pytest_plugins = "elasticai.creator.testing.cocotb_pytest"


@pytest.fixture
def translate():
    translation_pass = ir.Ir2Vhdl()
    loader = ir.PluginLoader(translation_pass)
    loader.load_from_package("xnor_popcount_mac")

    def _translate(g: ir.DataGraph):
        return translation_pass(g, Registry())

    return _translate


def test_can_load_into_translation_pass(translate):
    generated_name = "my_mac"
    assert f"{generated_name}.vhd" in set(
        code[0]
        for code in translate(
            factory.graph(
                attribute(weight="1001", kernel_size=2, parallelism=1),
                type="binary_filter",
                name=generated_name,
            ),
        )
    )


def test_binary_filter_rejects_incompatible_weight_and_kernel_size(translate):
    with pytest.raises(ValueError, match="divisible by kernel_size"):
        list(
            translate(
                factory.graph(
                    attribute(weight="101", kernel_size=2, parallelism=1),
                    type="binary_filter",
                    name="invalid_binary_filter",
                ),
            )
        )


@cocotb.test()
@eai_testbench
async def check_binary_filter(dut, weight, input, expected, parallelism):
    cocotb.start_soon(Clock(dut.clk, 10, "ns").start())
    stream = StreamInterface.from_dut(dut)
    reset = ClockReset.from_dut(dut)
    dut.src_valid.value = 0
    dut.dst_ready.value = 0
    await RisingEdge(dut.clk)
    await reset.reset_active_high()
    dut.en.value = 1
    collect_task = cocotb.start_soon(
        stream.collect_chunks(expected_count=1, max_cycles=10)
    )
    await stream.drive_chunks([input])
    observed = await collect_task
    assert observed == [expected]


@pytest.mark.parametrize(
    ["weight", "input", "expected", "parallelism"],
    [("1001", "0101", "1", 1)],
)
def test_binary_filter(
    cocotb_test_fixture: CocotbTestFixture,
    translate,
    weight,
    input,
    expected,
    parallelism,
):
    artifact_dir = cocotb_test_fixture.get_artifact_dir()
    ir2vhd_build_dir = artifact_dir / "vhdl"
    ir2vhd_build_dir.mkdir(exist_ok=True)
    resulting_code = translate(
        factory.graph(
            attribute(weight=f'"{weight}"', kernel_size=len(input), parallelism=1),
            type="binary_filter",
            name="binary_filter",
        ),
    )
    store_code(resulting_code, ir2vhd_build_dir)

    cocotb_test_fixture.set_srcs(collect_all_srcs_from_build_dir(ir2vhd_build_dir))
    cocotb_test_fixture.run(
        params=dict(),
        defines={},
    )


def store_code(code, build_dir):
    for name, lines in code:
        with open(build_dir / name, "w") as f:
            f.writelines(lines)


def collect_all_srcs_from_build_dir(build_dir):
    all_files = []
    for f in build_dir.iterdir():
        if f.is_file() and f.name.endswith("vhd"):
            all_files.append(f)
    return all_files


def xnor_popcount_kernel(weight: str, input: str) -> str:
    num_out_channels = len(weight) // len(input)
    kernel_size = len(input)

    def computation_per_out_channel(weight):
        xnored = []
        for left, right in zip(weight, input):
            if left == right:
                xnored.append("1")
            else:
                xnored.append("0")
        pop_count = 0
        for bit in xnored:
            if bit == "1":
                pop_count += 1
        integer_result = 2 * pop_count - len(weight)
        if integer_result >= 0:
            return "1"
        else:
            return "0"

    total = []
    for i in range(num_out_channels):
        total.append(
            computation_per_out_channel(weight[i * kernel_size : (i + 1) * kernel_size])
        )
    return "".join(total)
