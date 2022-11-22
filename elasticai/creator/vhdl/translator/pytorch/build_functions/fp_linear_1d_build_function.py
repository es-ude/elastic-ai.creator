import torch

from elasticai.creator.vhdl.quantized_modules.linear import FixedPointLinear
from elasticai.creator.vhdl.translator.abstract.layers import FPLinear1dModule


def build_fp_linear_1d(linear: FixedPointLinear, layer_id: str) -> FPLinear1dModule:
    def to_list(tensor: torch.Tensor) -> list:
        return tensor.detach().numpy().tolist()

    return FPLinear1dModule(
        layer_id=layer_id,
        weight=to_list(linear.weight),
        bias=to_list(linear.bias),
    )
