from io import StringIO
from unittest import TestCase

from elasticai.creator.integrationTests.vhdl.vhd_file_reader import (
    VHDLFileReaderWithoutComments,
)
from elasticai.creator.resource_utils import get_file
from elasticai.creator.vhdl.number_representations import (
    ClippedFixedPoint,
)
from vhdl.hw_equivalent_layers import (
    RootModule,
    FixedPointLinear,
    FixedPointHardSigmoid,
)


class FirstModel(RootModule):
    def __init__(self):
        super().__init__()
        self.data_width = 16
        fp_factory = ClippedFixedPoint.get_factory(
            total_bits=self.data_width, frac_bits=8
        )
        self.fp_linear = FixedPointLinear(
            in_features=1,
            out_features=1,
            fixed_point_factory=fp_factory,
            data_width=self.data_width,
        )
        self.fp_hard_sigmoid = FixedPointHardSigmoid(
            fixed_point_factory=fp_factory, data_width=self.data_width
        )

    def forward(self, x):
        return self.hard_sigmoid(self.fp_linear(x))


class GenerateLinearHardSigmoidNetwork(TestCase):
    def setUp(self):
        self.model = FirstModel()
        with get_file(
            "elasticai.creator.integrationTests.vhdl", "expected_network.vhd"
        ) as f:
            self.expected_code = VHDLFileReaderWithoutComments(f).as_list()

    @staticmethod
    def extract_portmap(vhdl_module):
        lines_are_relevant = False
        for line in GenerateLinearHardSigmoidNetwork.get_network_vhdl_code(vhdl_module):
            if lines_are_relevant:
                if line == ");":
                    break
                else:
                    yield line
            elif line == "port (":
                lines_are_relevant = True

    @staticmethod
    def get_network_vhdl_code(module):
        vhdl_file = next(iter(module.files))
        code = "\n".join(vhdl_file.code())
        io = StringIO(code)
        code = VHDLFileReaderWithoutComments(io).as_list()
        return code

    def get_generated_portmap_signal_line(self, signal_id, key, value) -> str:
        self.model.elasticai_tags.update({f"{key}": value})
        portmap = self.extract_portmap(self.model.translate())
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

    def test_check_network_vhdl_file(self):
        self.model.elasticai_tags.update(
            {
                "x_address_width": 1,
                "y_address_width": 1,
                "data_width": 16,
            }
        )
        self.assertEqual(
            list(self.expected_code),
            list(self.get_network_vhdl_code(self.model.translate())),
        )
