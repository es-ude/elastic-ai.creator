from dataclasses import dataclass

from elasticai.creator.vhdl.number_representations import FixedPoint, FixedPointConfig
from elasticai.creator.vhdl.templates import VHDLTemplate


@dataclass
class FPHardSigmoidComponent:
    layer_id: str
    zero_threshold: FixedPoint
    one_threshold: FixedPoint
    slope: FixedPoint
    y_intercept: FixedPoint
    fixed_point_factory: FixedPointConfig

    @property
    def name(self) -> str:
        return f"fp_hard_sigmoid_{self.layer_id}.vhd"

    def lines(self) -> list[str]:
        code = VHDLTemplate(
            base_name="fp_hard_sigmoid",
            data_width=str(self.fixed_point_factory.total_bits),
            frac_width=str(self.fixed_point_factory.frac_bits),
            one=str(self.fixed_point_factory(1).to_signed_int()),
            zero_threshold=str(self.zero_threshold.to_signed_int()),
            one_threshold=str(self.one_threshold.to_signed_int()),
            y_intercept=str(self.y_intercept.to_signed_int()),
            slope=str(self.slope.to_signed_int()),
            layer_name=self.layer_id,
        )

        return code.lines()
