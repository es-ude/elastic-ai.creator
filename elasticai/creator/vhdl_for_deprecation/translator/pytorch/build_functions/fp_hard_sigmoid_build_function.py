from elasticai.creator.nn.hard_sigmoid import HardSigmoid
from elasticai.creator.vhdl.number_representations import FixedPointConfig
from elasticai.creator.vhdl_for_deprecation.translator.abstract.layers import (
    FPHardSigmoidModule,
)


def build_fp_hard_sigmoid(
    layer: HardSigmoid, layer_id: str, fixed_point_factory: FixedPointConfig
) -> FPHardSigmoidModule:
    return FPHardSigmoidModule(
        layer_id=layer_id, fixed_point_factory=fixed_point_factory
    )
