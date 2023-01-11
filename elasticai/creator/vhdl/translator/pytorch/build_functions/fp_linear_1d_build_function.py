import torch

from elasticai.creator.nn.linear import FixedPointLinear
from elasticai.creator.vhdl.number_representations import FixedPointFactory
from elasticai.creator.vhdl.translator.abstract.layers import FPLinear1dModule


def build_fp_linear_1d(
    layer: FixedPointLinear,
    layer_id: str,
    fixed_point_factory: FixedPointFactory,
    work_library_name: str,
) -> FPLinear1dModule:
    def to_list(tensor: torch.Tensor) -> list:
        return tensor.detach().numpy().tolist()

    return FPLinear1dModule(
        layer_id=layer_id,
        weight=to_list(layer.weight),
        bias=to_list(layer.bias),
        fixed_point_factory=fixed_point_factory,
        work_library_name=work_library_name,
    )
