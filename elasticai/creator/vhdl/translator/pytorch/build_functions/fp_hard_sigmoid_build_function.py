from elasticai.creator.vhdl.quantized_modules.hard_sigmoid import FixedPointHardSigmoid
from elasticai.creator.vhdl.translator.abstract.layers.fp_hard_sigmoid_module import (
    FPHardSigmoidModule,
)


def build_fp_hard_sigmoid(hard_sigmoid: FixedPointHardSigmoid) -> FPHardSigmoidModule:
    return FPHardSigmoidModule()
