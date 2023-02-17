from elasticai.creator.vhdl.number_representations import FixedPoint, FixedPointConfig
from elasticai.creator.vhdl.templates.vhdl_template import VHDLTemplate


class FPHardSigmoidFile(VHDLTemplate):
    def __init__(
        self,
        layer_id: str,
        zero_threshold: FixedPoint,
        one_threshold: FixedPoint,
        slope: FixedPoint,
        y_intercept: FixedPoint,
        fixed_point_factory: FixedPointConfig,
    ):
        d = dict(
            data_width=fixed_point_factory.total_bits,
            frac_width=fixed_point_factory.frac_bits,
            one=fixed_point_factory(1).to_signed_int(),
            zero_threshold=zero_threshold.to_signed_int(),
            one_threshold=one_threshold.to_signed_int(),
            y_intercept=y_intercept.to_signed_int(),
            slope=slope.to_signed_int(),
            layer_name=layer_id,
        )
        stringified_d = dict(((k, str(v)) for k, v in d.items()))
        name = "fp_hard_sigmoid"
        super().__init__(template_name=name, **stringified_d)
