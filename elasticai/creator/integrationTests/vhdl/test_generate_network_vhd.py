from unittest import TestCase

from elasticai.creator.vhdl.modules import Module
from elasticai.creator.vhdl.number_representations import (
    ClippedFixedPoint,
)
from elasticai.creator.vhdl.quantized_modules import (
    FixedPointLinear,
    FixedPointHardSigmoid,
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
    @staticmethod
    def extract_portmap(model):
        vhdl_modules = model.to_vhdl()
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

    def check_output_address_width(self, value: int):
        model = FirstModel()
        model.elasticai_tags.update({"output_address_width": value})
        expected = f"output_address  : in std_logic_vector({value}-1 downto 0);"
        portmap = self.extract_portmap(model)
        actual = next(filter(lambda line: line.startswith("output_address"), portmap))
        self.assertEqual(expected, actual)

    def test_portmap_output_addr_width_is_4(
        self,
    ):
        self.check_output_address_width(4)

    def test_portmap_output_addr_width_is_8(self):
        self.check_output_address_width(8)
