import unittest
from typing import Iterable

from elasticai.creator.tests.code_utilities_for_testing import VHDLCodeTestCase
from elasticai.creator.tests.integration.vhdl.models_for_testing import FirstModel

Code = list[str]


class GeneratedNetworkVHDMatchesTargetForSingleModelVersion(VHDLCodeTestCase):
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
        self.actual_code = self.unified_vhdl_from_module(self.model.translate())

    def test_port_def_matches_target(self):
        expected_port_def = self.code_section_from_string(
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
        actual = self.extract_section_from_code(
            begin="port (", end=");", lines=self.actual_code
        )
        self.check_lines_are_equal_ignoring_order(expected_port_def, actual)

    @unittest.skip
    def test_signal_defs_match_target(self):
        actual_code = self.actual_code
        expected_signal_defs = VHDLCodeTestCase.code_section_from_string(
            """
    -- fp_linear
    signal fp_linear_enable : std_logic := '0';
    signal fp_linear_clock : std_logic := '0';
    signal fp_linear_done : std_logic := '0';
    signal fp_linear_x : std_logic_vector(15 downto 0) := (other => '0');
    signal fp_linear_y : std_logic_vector(15 downto 0) := (other => '0');
    signal fp_linear_x_address : std_logic_vector(0 downto 0) := (other => '0');
    signal fp_linear_y_address : std_logic_vector(0 downto 0) := (other => '0');

    -- fp_hard_sigmoid
    signal fp_hard_sigmoid_enable : std_logic := '0';
    signal fp_hard_sigmoid_clock : std_logic := '0';
    signal fp_hard_sigmoid_x : std_logic_vector(15 downto 0) := (other => '0');
    signal fp_hard_sigmoid_y : std_logic_vector(15 downto 0) := (other => '0');
            """
        )
        actual_sections = self.extract_section_from_code(
            begin="architecture rtl of fp_network is", end="begin", lines=actual_code
        )
        self.check_lines_are_equal_ignoring_order(expected_signal_defs, actual_sections)

    @unittest.skip
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
        portmaps = self.extract_section_from_code(
            begin="port map (", end=");", lines=self.actual_code
        )
        for counter, portmap in enumerate(portmaps):
            with self.subTest(f"portmap number {counter}"):
                with self.subTest("lines end with comma"):
                    self.assertTrue(all((line.endswith(",") for line in portmap[:-1])))
                with self.subTest("last line has no comma"):
                    self.assertFalse(portmap[-1].endswith(","))

    def check_portmaps(self, expected: str, portmap_start: str):
        expected_portmap: Iterable[Code] = VHDLCodeTestCase.code_section_from_string(
            expected
        )
        actual: Iterable[Code] = self.extract_section_from_code(
            begin=[portmap_start, "port map ("], end=");", lines=self.actual_code
        )
        expected_portmap = self.strip_comma_from_portmaps(expected_portmap)
        actual = self.strip_comma_from_portmaps(actual)
        self.check_lines_are_equal_ignoring_order(expected_portmap, actual)

    @unittest.skip
    def test_fp_hard_sigmoid_portmap_is_generated(self):
        self.check_portmaps(
            expected="""
        enable => fp_hard_sigmoid_enable,
        clock => fp_hard_sigmoid_clock,
        x => fp_hard_sigmoid_x,
        y => fp_hard_sigmoid_y""",
            portmap_start="fp_hard_sigmoid : entity work.fp_hard_sigmoid(rtl)",
        )
