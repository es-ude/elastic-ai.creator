from elasticai.creator.nn.relu import ReLU
from elasticai.creator.vhdl.number_representations import FixedPointConfig
from elasticai.creator.vhdl_for_deprecation.translator.abstract.layers.fp_relu_module import (
    FPReLUModule,
)


def build_fp_relu(
    layer: ReLU, layer_id: str, fixed_point_factory: FixedPointConfig
) -> FPReLUModule:
    return FPReLUModule(layer_id=layer_id, fixed_point_factory=fixed_point_factory)
