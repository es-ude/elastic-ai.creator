from elasticai.creator.vhdl.number_representations import FixedPoint, FixedPointFactory
from elasticai.creator.vhdl.vhdl_files import VHDLFile


class FPHardTanhComponent(VHDLFile):
    def __init__(
        self,
        min_val: FixedPoint,
        max_val: FixedPoint,
        fixed_point_factory: FixedPointFactory,
        layer_id: str,
    ):
        d = dict(
            data_width=fixed_point_factory.total_bits,
            frac_width=fixed_point_factory.frac_bits,
            min_val=min_val.to_signed_int(),
            max_val=max_val.to_signed_int(),
            layer_name=layer_id,
        )
        stringified_d = dict(((k, str(v)) for k, v in d.items()))
        super().__init__(name="fp_hard_tanh", **stringified_d)
