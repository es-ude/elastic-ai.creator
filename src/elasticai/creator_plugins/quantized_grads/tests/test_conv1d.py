from elasticai.creator_plugins.quantized_grads.base_modules import Conv1d
from elasticai.creator_plugins.quantized_grads.fixed_point import (
    FixedPointConfigV2,
    QuantizeForwHTE,
    QuantizeParamToFixedPointHTE,
    QuantizeParamToFixedPointStochastic,
)


def test_conv1d_fxp_init():
    conf = FixedPointConfigV2(8, 3)

    Conv1d(
        QuantizeForwHTE(conf),
        QuantizeParamToFixedPointHTE(conf),
        3,
        2,
        2,
        bias_quantization=QuantizeParamToFixedPointStochastic(conf),
    )
