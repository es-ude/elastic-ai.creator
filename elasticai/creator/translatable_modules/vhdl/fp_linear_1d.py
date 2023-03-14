from elasticai.creator.hdl.design_base.design import Design
from elasticai.creator.hdl.vhdl.designs.fp_linear_1d import FPLinear1d as FPLinearDesign
from elasticai.creator.nn.linear import FixedPointLinear


class FPLinear1d(FixedPointLinear):
    def translate(self) -> Design:
        return FPLinearDesign(
            frac_bits=self.frac_bits,
            total_bits=self.total_bits,
            in_feature_num=self.in_features,
            out_feature_num=self.out_features,
        )
