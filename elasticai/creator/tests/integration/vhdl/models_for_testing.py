from elasticai.creator.vhdl.number_representations import ClippedFixedPoint
from elasticai.creator.vhdl.translatable_modules.layers import (
    FixedPointHardSigmoid,
    FixedPointLinear,
    RootModule,
)


class FirstModel(RootModule):
    def __init__(self):
        super().__init__()
        self.data_width = 16
        fp_factory = ClippedFixedPoint.get_builder(
            total_bits=self.data_width, frac_bits=8
        )
        self.fp_linear = FixedPointLinear(
            in_features=2,
            out_features=2,
            fixed_point_factory=fp_factory,
            data_width=self.data_width,
        )
        self.fp_hard_sigmoid = FixedPointHardSigmoid(
            fixed_point_factory=fp_factory, data_width=self.data_width
        )

    def forward(self, x):
        return self.fp_hard_sigmoid(self.fp_linear(x))
