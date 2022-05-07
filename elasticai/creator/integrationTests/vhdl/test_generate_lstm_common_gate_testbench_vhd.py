import pathlib

from elasticai.creator.vhdl.generator.lstm_testbench_generator import (
    LSTMCommonGateTestBench,
)
from elasticai.creator.tests.vhdl.vhdl_file_testcase import GeneratedVHDLCodeTest


class LSTMCommonGateTestBenchTest(GeneratedVHDLCodeTest):
    def test_compare_files(self) -> None:
        with open(pathlib.Path(__file__).parent.resolve().joinpath("expected_common_gate_testbench.vhd"), 'r') as f:
            expected_code = f.read()

        lstm_common_gate = LSTMCommonGateTestBench(
            data_width=16,
            frac_width=8,
            vector_len_width=4,
            component_name="lstm_common_gate",
            x_mem_list_for_testing=[
                'x"0013",x"0000",x"0010",x"0013",x"000c",x"0005",x"0005",x"0013",x"0004",x"0002"',
                'x"0014",x"000d",x"0017",x"0008",x"0002",x"0007",x"0002",x"0015",x"0001",x"0010"',
                'x"000f",x"0017",x"000d",x"000f",x"0001",x"0009",x"0002",x"0007",x"0008",x"0013"',
                'x"000c",x"0007",x"0001",x"0019",x"0008",x"000c",x"0019",x"000b",x"0008",x"000d"',
                'x"0005",x"0013",x"0002",x"0013",x"000c",x"000f",x"0003",x"0004",x"0010",x"0001"',
            ],
            w_mem_list_for_testing=[
                'x"0011",x"0018",x"0000",x"000d",x"0014",x"000f",x"0012",x"0007",x"0017",x"0012"',
                'x"000e",x"0014",x"0005",x"0015",x"0009",x"0013",x"0007",x"0016",x"0008",x"0004"',
                'x"0001",x"000a",x"0008",x"0010",x"0008",x"0001",x"0016",x"0013",x"0016",x"000a"',
                'x"000e",x"0015",x"0001",x"000b",x"0014",x"0012",x"000f",x"0000",x"0008",x"000e"',
                'x"0006",x"000d",x"0005",x"0009",x"0017",x"0017",x"000e",x"000d",x"0000",x"0019"',
            ],
            b_list_for_testing=['x"008a"', 'x"0064"', 'x"009b"', 'x"004c"', 'x"0092"'],
            y_list_for_testing=[142, 105, 159, 82, 150],
        )
        lstm_common_gate_code = lstm_common_gate()
        lstm_common_gate_code_str = ""
        for line in lstm_common_gate_code:
            lstm_common_gate_code_str += line + "\n"
        self.check_generated_code(expected_code, lstm_common_gate_code_str)
