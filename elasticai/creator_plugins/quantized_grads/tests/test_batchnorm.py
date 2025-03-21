from elasticai.creator_plugins.quantized_grads.base_modules import BatchNorm2d
from elasticai.creator_plugins.quantized_grads.fixed_point import (
    FixedPointConfigV2,
    QuantizeForwHTE,
    QuantizeParamToFixedPointHTE,
    QuantizeParamToFixedPointStochastic,
)


def test_batchnorm_fxp_init():
    conf = FixedPointConfigV2(8, 3)

    BatchNorm2d(
        math_ops=QuantizeForwHTE(forward_conf=conf),
        weight_quantization=QuantizeParamToFixedPointHTE(conf),
        bias_quantization=QuantizeParamToFixedPointStochastic(conf),
        num_features=3,
    )
