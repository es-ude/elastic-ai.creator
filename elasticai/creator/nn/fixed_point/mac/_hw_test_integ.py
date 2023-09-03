import glob

import torch
from fixed_point.mac._signal_number_converter import SignalNumberConverter
from fixed_point.mac.sw_function import MacLayer
from fixed_point.mac.testbench import TestBench

from elasticai.creator.file_generation.on_disk_path import OnDiskPath
from elasticai.creator.file_generation.template import InProjectTemplate
from elasticai.creator.vhdl.test_bench_runner import TestBenchRunner

"""
Notes:
  - The software layer knows the correct result for an input
  - The translatable layer knows how to convert inputs and outputs to correct vhdl representation
"""
"""
TODO:
 - add automatic output parsing and compare to expected result
"""


def hw_test():
    root_name = "hw_test"
    root = OnDiskPath(root_name)
    testbench_name = "testbench_fxp_mac"
    x1 = (0.0, 1.0)
    x2 = (1.0, -1.0)
    mac = MacLayer(total_bits=4, frac_bits=2)
    y = mac(torch.tensor(x1), torch.tensor(x2))
    testbench = TestBench(
        total_bits=4, frac_bits=2, inputs=zip(x1, x2), name="mac_fxp_test_bench"
    )
    mac_design = mac.create_design()
    testbench.save_to(root)
    mac_design.save_to(root)
    files = list(glob.glob(f"**/*.vhd", root_dir=root_name, recursive=True))
    runner = TestBenchRunner(
        workdir=f"{root_name}", files=files, test_bench_name=testbench_name
    )
    runner.initialize()
    runner.run()


if __name__ == "__main__":
    hw_test()
