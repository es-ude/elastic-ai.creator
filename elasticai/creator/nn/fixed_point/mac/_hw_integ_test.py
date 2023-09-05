import glob
from functools import partial

import pytest
import torch

from elasticai.creator.file_generation.on_disk_path import OnDiskPath
from elasticai.creator.nn.fixed_point.mac.number_conversion import (
    bits_to_rational,
    rational_to_bits,
)
from elasticai.creator.vhdl.test_bench_runner import TestBenchRunner

from .sw_function import MacLayer
from .testbench import TestBench


@pytest.mark.simulation
def test_mac_hw():
    root_name = "hw_test"
    root = OnDiskPath(root_name)
    x1 = (0.0, 1.0)
    x2 = (0.0, -1.0)
    mac = MacLayer(total_bits=4, frac_bits=2)
    test_bench_name = "testbench_fxp_mac"
    y = mac(torch.tensor(x1), torch.tensor(x2)).item()

    def prepare_inputs_for_test_bench(x1, x2):
        print(x1)
        convert = partial(rational_to_bits, total_bits=4, frac_bits=2)
        x1 = map(convert, x1)
        x2 = map(convert, x2)
        inputs = {"x1": ", ".join([f'b"{x}"' for x in x1])}
        inputs.update({"x2": ", ".join([f'b"{x}"' for x in x2])})
        print(inputs)
        return inputs

    testbench = TestBench(
        total_bits=4,
        frac_bits=2,
        inputs=prepare_inputs_for_test_bench(x1, x2),
        name=test_bench_name,
    )
    mac_design = mac.create_design()
    testbench.save_to(root)
    mac_design.save_to(root)
    files = list(glob.glob(f"**/*.vhd", root_dir=root_name, recursive=True))
    runner = TestBenchRunner(
        workdir=f"{root_name}", files=files, test_bench_name=test_bench_name
    )
    runner.initialize()
    runner.run()
    actual = runner.getReportedContent()[0]
    actual = bits_to_rational(actual, frac_bits=2)
    assert y == actual
