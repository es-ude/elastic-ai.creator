from io import StringIO
from unittest import TestCase

from elasticai.creator.integrationTests.vhdl.vhd_file_reader import (
    VHDFileReaderWithoutComments,
)
from elasticai.creator.vhdl.modules import Module
from elasticai.creator.resource_utils import read_text, get_file
from elasticai.creator.vhdl.number_representations import (
    ClippedFixedPoint,
)
from elasticai.creator.vhdl.quantized_modules import (
    FixedPointLinear,
    FixedPointHardSigmoid,
)


class GenerateNetworkTestTestCase(TestCase):
    def test_network_vhd_for_network_with_linear_and_hardsigmoid_layer_is_correctly_generated(
        self,
    ):
        class MyModel(Module):
            def __init__(self):
                super().__init__()
                fp_factory = ClippedFixedPoint.get_factory(total_bits=16, frac_bits=8)
                self.fp_linear = FixedPointLinear(
                    in_features=1, out_features=1, fixed_point_factory=fp_factory
                )
                self.hard_sigmoid = FixedPointHardSigmoid(
                    fixed_point_factory=fp_factory
                )

            def forward(self, x):
                return self.hard_sigmoid(self.fp_linear(x))

        model = MyModel()
        expected = tuple(
            VHDFileReaderWithoutComments(
                open(
                    get_file(
                        "elasticai.creator.integrationTests.vhdl",
                        "expected_network.vhd",
                    ),
                    "r",
                )
            )
        )
        vhdl_modules = model.to_vhdl()
        network_vhdl_module = next(
            filter(lambda module: module.name == "network", vhdl_modules)
        )
        network_vhd_file = next(iter(network_vhdl_module.files))
        print(network_vhd_file.code)

        actual = tuple(network_vhd_file.code)
        self.assertEqual(expected, actual)
