from elasticai.creator.vhdl.quantized_modules.relu import FixedPointReLU
from elasticai.creator.vhdl.translator.abstract.layers.fp_relu_module import (
    FPReLUModule,
)


def build_fp_relu(fp_relu: FixedPointReLU) -> FPReLUModule:
    return FPReLUModule()
