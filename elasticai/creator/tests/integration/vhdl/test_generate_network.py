from typing import Iterable
from unittest import TestCase

from elasticai.creator.resource_utils import get_file
from elasticai.creator.tests.integration.vhdl.code_test_case import CodeTestCase
from elasticai.creator.tests.integration.vhdl.vhd_file_reader import (
    VHDLFileReaderWithoutComments,
)
from elasticai.creator.tests.text_parsing import extract_section
from elasticai.creator.vhdl.code import Code
from elasticai.creator.vhdl.hw_equivalent_layers.layers import (
    FixedPointHardSigmoid,
    FixedPointLinear,
    RootModule,
)
from elasticai.creator.vhdl.number_representations import ClippedFixedPoint

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
        self.fp_hard_sigmoid = FixedPointHardSigmoid(data_width=self.data_width)

    def forward(self, x):
        return self.fp_hard_sigmoid(self.fp_linear(x))


class GenerateNetworkRootFileFromDifferentModelVersions(TestCase):
    def setUp(self):
        self.model = FirstModel()

        with get_file(
            "elasticai.creator.tests.integration.vhdl", "expected_network.vhd"
        ) as f:
            self.expected_code = VHDLFileReaderWithoutComments(f).as_list()

    def get_generated_portmap_signal_line(self, signal_id, value) -> str:
        self.model.elasticai_tags.update({f"{signal_id}_width": value})
        code = CodeTestCase.unified_vhdl_from_module(self.model.translate())
        portmap = extract_section(begin="port (", end=");", lines=code)[0]
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


class GeneratedNetworkVHDMatchesTargetForSingleModelVersion(CodeTestCase):
    """
    Tests:
      - each of the port maps is generated correctly
      - signals for each layer are connected correctly
    """

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
        self.actual_code = CodeTestCase.unified_vhdl_from_module(self.model.translate())

    def test_port_def_matches_target(self):
        expected_port_def = CodeTestCase.code_section_from_string(
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
        actual = extract_section(begin="port (", end=");", lines=self.actual_code)
        self.check_lines_are_equal_ignoring_order(expected_port_def, actual)

    def test_signal_defs_match_target(self):
        actual_code = self.actual_code
        expected_signal_defs = CodeTestCase.code_section_from_string(
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
        actual_sections = extract_section(
            begin="architecture rtl of fp_network is", end="begin", lines=actual_code
        )
        self.check_lines_are_equal_ignoring_order(expected_signal_defs, actual_sections)

    def test_fp_linear_portmap_is_generated(self):
        self.check_portmaps(
            expected="""
        enable => fp_linear_enable,
        clock => fp_linear_clock,

        x => fp_linear_x,
        y => fp_linear_y,
        x_address => fp_linear_x_address,
        y_address => fp_linear_y_address,
        done => fp_linear_done
        """,
            portmap_start="fp_linear : entity work.fp_linear(rtl)",
        )

    @staticmethod
    def strip_comma_from_portmaps(portmaps: Iterable[Code]) -> Iterable[Code]:
        for portmap in portmaps:
            yield (line.rstrip(",") for line in portmap)

    def test_all_but_last_portmap_line_have_trailing_comma(self):
        portmaps = extract_section(begin="port map(", end=");", lines=self.actual_code)
        for counter, portmap in enumerate(portmaps):
            with self.subTest(f"portmap number {counter}"):
                with self.subTest("lines end with comma"):
                    self.assertTrue(all((line.endswith(",") for line in portmap[:-1])))
                with self.subTest("last line has no comma"):
                    self.assertFalse(portmap[-1].endswith(","))

    def check_portmaps(self, expected: str, portmap_start: str):
        expected_portmap: Iterable[Code] = CodeTestCase.code_section_from_string(
            expected
        )
        actual: Iterable[Code] = extract_section(
            begin=[portmap_start, "port map("], end=");", lines=self.actual_code
        )
        expected_portmap = self.strip_comma_from_portmaps(expected_portmap)
        actual = self.strip_comma_from_portmaps(actual)
        self.check_lines_are_equal_ignoring_order(expected_portmap, actual)

    def test_fp_hard_sigmoid_portmap_is_generated(self):
        self.check_portmaps(
            expected="""
        enable => fp_hard_sigmoid_enable,
        clock => fp_hard_sigmoid_clock,
        x => fp_hard_sigmoid_x,
        y => fp_hard_sigmoid_y""",
            portmap_start="fp_hard_sigmoid : entity work.fp_hard_sigmoid(rtl)",
        )


class SignalConnectionsTest(CodeTestCase):
    def setUp(self) -> None:
        self.model = FirstModel()
        self.model.elasticai_tags.update(
            {
                "x_address_width": 1,
                "y_address_width": 1,
                "y_width": 16,
                "x_width": 16,
            }
        )
        code = CodeTestCase.unified_vhdl_from_module(self.model.translate())
        self.actual_connections: Code = extract_section(
            begin="begin",
            end="fp_linear : entity work.fp_linear(rtl)",
            lines=code,
        )[0]

    def test_x_is_connected_to_fp_linear_x(self):
        self.check_contains_all_expected_lines(
            expected=["fp_linear_x <= x;"], actual=self.actual_connections
        )
