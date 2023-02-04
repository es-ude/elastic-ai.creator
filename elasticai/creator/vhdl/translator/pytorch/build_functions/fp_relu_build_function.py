from elasticai.creator.nn.relu import ReLU
from elasticai.creator.vhdl.number_representations import FixedPointFactory
from elasticai.creator.vhdl.translator.abstract.layers.fp_relu_module import (
    FPReLUModule,
)


def build_fp_relu(
    layer: ReLU, layer_id: str, fixed_point_factory: FixedPointFactory
) -> FPReLUModule:
    return FPReLUModule(layer_id=layer_id, fixed_point_factory=fixed_point_factory)
