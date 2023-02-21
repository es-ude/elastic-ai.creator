from elasticai.creator.nn.hard_tanh import HardTanh
from elasticai.creator.vhdl.number_representations import FixedPointConfig
from elasticai.creator.vhdl_for_deprecation.translator.abstract.layers.fp_hard_sigmoid_module import (
    FPHardSigmoidModule,
)


def build_fp_hard_tanh(
    layer: HardTanh, layer_id: str, fixed_point_factory: FixedPointConfig
) -> FPHardSigmoidModule:
    return FPHardSigmoidModule(
        layer_id=layer_id, fixed_point_factory=fixed_point_factory
    )
