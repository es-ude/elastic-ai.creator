import torch

from elasticai.creator.vhdl.quantized_modules.linear import FixedPointLinear
from elasticai.creator.vhdl.translator.abstract.layers import FPLinear1dModule


def build_fp_linear_1d(linear: FixedPointLinear) -> FPLinear1dModule:
    def to_list(tensor: torch.Tensor) -> list:
        return tensor.detach().numpy().tolist()

    return FPLinear1dModule(
        layer_name=linear.layer_name,
        weight=to_list(linear.weight),
        bias=to_list(linear.bias),
    )
