import unittest
from unittest import TestCase

from elasticai.creator.vhdl.number_representations import (
    ClippedFixedPoint,
)
from elasticai.creator.vhdl.modules import (
    FixedPointLinear,
    FixedPointHardSigmoid,
    Module,
)


class FirstModel(Module):
    def __init__(self):
        super().__init__()
        fp_factory = ClippedFixedPoint.get_factory(total_bits=16, frac_bits=8)
        self.fp_linear = FixedPointLinear(
            in_features=1, out_features=1, fixed_point_factory=fp_factory
        )
        self.hard_sigmoid = FixedPointHardSigmoid(fixed_point_factory=fp_factory)

    def forward(self, x):
        return self.hard_sigmoid(self.fp_linear(x))


class GenerateLinearHardSigmoidNetwork(TestCase):
    def setUp(self):
        self.model = FirstModel()

    @staticmethod
    def extract_portmap(vhdl_modules):
        network_vhdl_module = next(
            filter(lambda module: module.name == "network", vhdl_modules)
        )
        network_vhd_file = next(iter(network_vhdl_module.files))
        lines_are_relevant = False
        for line in network_vhd_file.code:
            if lines_are_relevant:
                if line.lstrip() == ");":
                    break
                else:
                    yield line.lstrip()
            elif line.lstrip() == "port (":
                lines_are_relevant = True

    def get_generated_portmap_signal_line(self, signal_id, key, value) -> str:
        self.model.elasticai_tags.update({f"{key}": value})
        portmap = self.extract_portmap(self.model.to_vhdl())
        actual = next(filter(lambda line: line.startswith(f"{signal_id}:"), portmap))
        return actual

    def check_y_address_width(self, value: int):
        actual = self.get_generated_portmap_signal_line(
            "y_address", "y_address_width", value
        )
        self.assertEqual(f"y_address: in std_logic_vector({value}-1 downto 0);", actual)

    def check_x_data_width(self, value: int):
        actual = self.get_generated_portmap_signal_line("x", "data_width", value)
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
        actual = self.get_generated_portmap_signal_line("y", "data_width", 8)
        self.assertEqual("y: out std_logic_vector(8-1 downto 0);", actual)

    def test_x_address_width_is_16(self):
        actual = self.get_generated_portmap_signal_line(
            "x_address", "x_address_width", 16
        )
        self.assertEqual("x_address: out std_logic_vector(16-1 downto 0);", actual)
