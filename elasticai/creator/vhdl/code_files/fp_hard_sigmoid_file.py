from elasticai.creator.vhdl.number_representations import (
    FixedPoint,
    FixedPointFactory,
)
from vhdl.vhdl_files import VHDLFile


class FPHardSigmoidComponent(VHDLFile):
    def __init__(
        self,
        layer_id: int,
        zero_threshold: FixedPoint,
        one_threshold: FixedPoint,
        slope: FixedPoint,
        y_intercept: FixedPoint,
        fixed_point_factory: FixedPointFactory,
    ):
        data_width = str(fixed_point_factory.total_bits)
        frac_width = str(fixed_point_factory.frac_bits)
        one = str(fixed_point_factory(1).to_signed_int())
        zero_threshold = str(zero_threshold.to_signed_int())
        one_threshold = str(one_threshold.to_signed_int())
        y_intercept = str(y_intercept.to_signed_int())
        slope = str(slope.to_signed_int())
        name = "fp_hard_sigmoid"
        super().__init__(
            name=name,
            parameters=dict(
                layer_id=f"{name}_{layer_id}",
                data_width=data_width,
                frac_width=frac_width,
                one=one,
                zero_threshold=zero_threshold,
                fixed_point_factory=fixed_point_factory,
                one_threshold=one_threshold,
                slope=slope,
                y_intercept=y_intercept,
            ),
        )
