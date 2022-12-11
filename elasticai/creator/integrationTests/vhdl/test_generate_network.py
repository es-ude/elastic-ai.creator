import unittest
from io import StringIO
from unittest import TestCase

from elasticai.creator.integrationTests.vhdl.vhd_file_reader import (
    VHDLFileReaderWithoutComments,
)
from elasticai.creator.resource_utils import get_file
from elasticai.creator.vhdl.number_representations import (
    ClippedFixedPoint,
)
from elasticai.creator.vhdl.hw_equivalent_layers import (
    RootModule,
    FixedPointLinear,
    FixedPointHardSigmoid,
)
from vhdl.code import Code

"""
Tests:
    - Names of generated CodeModules are converted torch module_hierarchy names.
    - For every leaf module in our torch module we generate a CodeModule
    - For every node in our graph from tracing torch module we instantiate a CodeModule of the correct name in our network.vhd
"""


class FirstModel(RootModule):
    def __init__(self):
        super().__init__()
        self.data_width = 16
        fp_factory = ClippedFixedPoint.get_factory(
            total_bits=self.data_width, frac_bits=8
        )
        self.fp_linear = FixedPointLinear(
            in_features=2,
            out_features=2,
            fixed_point_factory=fp_factory,
            data_width=self.data_width,
        )
        self.fp_hard_sigmoid = FixedPointHardSigmoid(
            fixed_point_factory=fp_factory, data_width=self.data_width
        )

    def forward(self, x):
        return self.fp_hard_sigmoid(self.fp_linear(x))


def extract_port_lines(lines) -> list[str]:
    return extract_section(begin="port (", end=");", lines=lines)[0]


def extract_section(begin: str, end: str, lines: Code) -> list[list[str]]:
    extract = False
    content = []
    current_section = []
    for line in lines:
        if line == begin:
            extract = True
            continue
        elif extract and line == end:
            extract = False
            content.append(current_section)
            current_section = []
        elif extract:
            current_section.append(line)

    if extract:
        raise Exception(f"reached end of code before end: {end}")
    return content


def get_network_vhdl_code(module):
    vhdl_file = next(iter(module.files))
    code = "\n".join(vhdl_file.code())
    io = StringIO(code)
    code = VHDLFileReaderWithoutComments(io).as_list()
    return code


def code_from_string(s: str) -> Code:
    return VHDLFileReaderWithoutComments(StringIO(s))


class GenerateNetworkRootFileFromDifferentModelVersions(TestCase):
    def setUp(self):
        self.model = FirstModel()

        with get_file(
            "elasticai.creator.integrationTests.vhdl", "expected_network.vhd"
        ) as f:
            self.expected_code = VHDLFileReaderWithoutComments(f).as_list()

    def get_generated_portmap_signal_line(self, signal_id, value) -> str:
        self.model.elasticai_tags.update({f"{signal_id}_width": value})
        code = get_network_vhdl_code(self.model.translate())
        portmap = extract_port_lines(code)
        actual = next(filter(lambda line: line.startswith(f"{signal_id}:"), portmap))
        return actual

    def check_y_address_width(self, value: int):
        actual = self.get_generated_portmap_signal_line("y_address", value)
        self.assertEqual(f"y_address: in std_logic_vector({value}-1 downto 0);", actual)

    def check_x_data_width(self, value: int):
        actual = self.get_generated_portmap_signal_line("x", value)
        self.assertEqual(f"x: in std_logic_vector({value}-1 downto 0);", actual)

    def test_portmap_output_addr_width_is_4(
        self,
    ):
        self.check_y_address_width(4)

    def test_portmap_output_addr_width_is_8(self):
        self.check_y_address_width(8)

    def test_x_data_width_is_4(self):
        self.check_x_data_width(4)

    def test_y_data_width_is_8(self):
        actual = self.get_generated_portmap_signal_line("y", 8)
        self.assertEqual("y: out std_logic_vector(8-1 downto 0);", actual)

    def test_x_address_width_is_16(self):
        actual = self.get_generated_portmap_signal_line("x_address", 16)
        self.assertEqual("x_address: out std_logic_vector(16-1 downto 0);", actual)


class GeneratedNetworkVHDMatchesTargetForSingleModelVersion(unittest.TestCase):
    def setUp(self):
        self.model = FirstModel()
        self.model.elasticai_tags.update(
            {
                "x_address_width": 1,
                "y_address_width": 1,
                "y_width": 16,
                "x_width": 16,
            }
        )
        self.actual_code = get_network_vhdl_code(self.model.translate())

    def test_port_def_matches_target(self):
        self.model.elasticai_tags.update(
            {
                "x_address_width": 1,
                "y_address_width": 1,
                "y_width": 16,
                "x_width": 16,
            }
        )
        actual_code = self.actual_code
        expected_port_def = list(
            code_from_string(
                """
        enable: in std_logic;
        clock: in std_logic;
        x_address: out std_logic_vector(1-1 downto 0);
        y_address: in std_logic_vector(1-1 downto 0);
        x: in std_logic_vector(16-1 downto 0);
        y: out std_logic_vector(16-1 downto 0);
        done: out std_logic
        """
            )
        )
        actual_port_def = extract_port_lines(actual_code)
        self.check_all_expected_lines_are_present(expected_port_def, actual_port_def)

    def check_all_expected_lines_are_present(self, expected: Code, actual: Code):
        expected = sorted(expected)
        actual = sorted(actual)
        self.assertEqual(list(expected), list(actual))

    def test_signal_defs_match_target(self):
        actual_code = self.actual_code
        expected_signal_defs = list(
            code_from_string(
                """
    -- fp_linear
    signal fp_linear_enable : std_logic := '0';
    signal fp_linear_clock : std_logic := '0';
    signal fp_linear_done : std_logic := '0';
    signal fp_linear_x : std_logic_vector(15 downto 0);
    signal fp_linear_y : std_logic_vector(15 downto 0);
    signal fp_linear_x_address : std_logic_vector(0 downto 0);
    signal fp_linear_y_address : std_logic_vector(0 downto 0);

    -- fp_hard_sigmoid
    signal fp_hard_sigmoid_enable : std_logic := '0';
    signal fp_hard_sigmoid_clock : std_logic := '0';
    signal fp_hard_sigmoid_x : std_logic_vector(15 downto 0);
    signal fp_hard_sigmoid_y : std_logic_vector(15 downto 0);
            """
            )
        )
        actual_sections = extract_section(
            begin="architecture rtl of fp_network is", end="begin", lines=actual_code
        )
        self.assertEqual(1, len(actual_sections))
        actual_signal_defs = actual_sections[0]
        self.check_all_expected_lines_are_present(
            expected_signal_defs, actual_signal_defs
        )
