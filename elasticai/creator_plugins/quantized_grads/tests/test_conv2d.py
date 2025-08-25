from elasticai.creator.arithmetic import FxpParams
from elasticai.creator_plugins.quantized_grads.base_modules import Conv2d
from elasticai.creator_plugins.quantized_grads.fixed_point import (
    QuantizeForwHTE,
    QuantizeParamToFixedPointHTE,
    QuantizeParamToFixedPointStochastic,
)


def test_conv1d_fxp_init():
    conf = FxpParams(total_bits=8, frac_bits=3)

    Conv2d(
        QuantizeForwHTE(conf),
        QuantizeParamToFixedPointHTE(conf),
        3,
        2,
        2,
        bias_quantization=QuantizeParamToFixedPointStochastic(conf),
    )
