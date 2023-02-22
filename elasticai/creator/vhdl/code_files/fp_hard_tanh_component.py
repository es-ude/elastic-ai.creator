from dataclasses import dataclass

from elasticai.creator.resource_utils import read_text
from elasticai.creator.vhdl.code import Code, CodeFile
from elasticai.creator.vhdl.number_representations import FixedPoint, FixedPointFactory
from elasticai.creator.vhdl.vhdl_files import VHDLFile, expand_template


@dataclass
class FPHardTanhComponent(CodeFile):
    min_val: FixedPoint
    max_val: FixedPoint
    fixed_point_factory: FixedPointFactory
    layer_id: str

    @property
    def name(self) -> str:
        return f"fp_hard_tanh_{self.layer_id}.vhd"

    def code(self) -> Code:
        template = read_text("elasticai.creator.vhdl.templates", "fp_hard_tanh.tpl.vhd")

        code = expand_template(
            template=template,
            data_width=str(self.fixed_point_factory.total_bits),
            frac_width=str(self.fixed_point_factory.frac_bits),
            min_val=str(self.min_val.to_signed_int()),
            max_val=str(self.max_val.to_signed_int()),
            layer_name=str(self.layer_id),
        )

        return code
