from elasticai.creator.nn.quantized_grads.base_modules import Conv2d
from elasticai.creator.nn.quantized_grads.fixed_point import (
    FixedPointConfigV2,
    QuantizeForwHTE,
    QuantizeParamToFixedPointHTE,
    QuantizeParamToFixedPointStochastic,
)


def test_conv1d_fxp_init():
    conf = FixedPointConfigV2(8, 3)

    Conv2d(
        QuantizeForwHTE(conf),
        QuantizeParamToFixedPointHTE(conf),
        3,
        2,
        2,
        bias_quantization=QuantizeParamToFixedPointStochastic(conf),
    )
