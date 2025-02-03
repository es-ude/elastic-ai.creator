from elasticai.creator.nn.quantized_grads.base_modules import Conv1d
from elasticai.creator.nn.quantized_grads.fixed_point import (
    FixedPointConfigV2, QuantizeParamToFixedPointHTE, QuantizeParamToFixedPointStochastic,
    QuantizeForwHTE
)


def test_conv1d_fxp_init():
    conf = FixedPointConfigV2(8, 3)

    l = Conv1d(QuantizeForwHTE(conf),
               QuantizeParamToFixedPointHTE(conf),
               3,
               2,
               2,
               bias_quantization=QuantizeParamToFixedPointStochastic(conf))


