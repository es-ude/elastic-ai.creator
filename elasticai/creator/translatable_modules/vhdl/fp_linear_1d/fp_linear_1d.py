from elasticai.creator.hdl.design_base.design import Design
from elasticai.creator.nn.linear import FixedPointLinear
from elasticai.creator.translatable_modules.vhdl.fp_linear_1d.fp_linear_1d_hw_design import (
    FPLinear1d as FPLinearDesign,
)


class FPLinear1d(FixedPointLinear):
    def translate(self) -> Design:
        return FPLinearDesign(
            frac_bits=self.frac_bits,
            total_bits=self.total_bits,
            in_feature_num=self.in_features,
            out_feature_num=self.out_features,
            weights=self.ops.quantize(self.weight).tolist(),
        )
