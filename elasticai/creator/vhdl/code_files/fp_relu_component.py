from dataclasses import dataclass

from elasticai.creator.resource_utils import read_text
from elasticai.creator.vhdl.code import Code, CodeFile
from elasticai.creator.vhdl.number_representations import FixedPointFactory
from elasticai.creator.vhdl.vhdl_files import expand_template


@dataclass
class FPReLUComponent(CodeFile):
    layer_id: str
    fixed_point_factory: FixedPointFactory

    def __post_init__(self) -> None:
        self.data_width = self.fixed_point_factory.total_bits
        self.frac_width = self.fixed_point_factory.frac_bits

    @property
    def name(self) -> str:
        return f"fp_relu_{self.layer_id}.vhd"

    def code(self) -> Code:
        template = read_text("elasticai.creator.vhdl.templates", "fp_relu.tpl.vhd")

        code = expand_template(
            template,
            layer_name=self.layer_id,
            data_width=str(self.data_width),
            clock_option="false",
        )

        return code
