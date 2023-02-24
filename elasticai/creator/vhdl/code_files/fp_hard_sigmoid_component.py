from dataclasses import dataclass

from elasticai.creator.resource_utils import read_text
from elasticai.creator.vhdl.code import Code, CodeFile
from elasticai.creator.vhdl.number_representations import FixedPoint, FixedPointFactory
from elasticai.creator.vhdl.vhdl_files import VHDLFile, expand_template


@dataclass
class FPHardSigmoidComponent(CodeFile):
    layer_id: str
    zero_threshold: FixedPoint
    one_threshold: FixedPoint
    slope: FixedPoint
    y_intercept: FixedPoint
    fixed_point_factory: FixedPointFactory

    @property
    def name(self) -> str:
        return f"fp_hard_sigmoid_{self.layer_id}.vhd"

    def code(self) -> Code:
        template = read_text(
            "elasticai.creator.vhdl.templates", "fp_hard_sigmoid.tpl.vhd"
        )

        code = expand_template(
            template=template,
            data_width=str(self.fixed_point_factory.total_bits),
            frac_width=str(self.fixed_point_factory.frac_bits),
            one=str(self.fixed_point_factory(1).to_signed_int()),
            zero_threshold=str(self.zero_threshold.to_signed_int()),
            one_threshold=str(self.one_threshold.to_signed_int()),
            y_intercept=str(self.y_intercept.to_signed_int()),
            slope=str(self.slope.to_signed_int()),
            layer_name=self.layer_id,
        )

        return code
