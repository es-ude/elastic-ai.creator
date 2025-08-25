from elasticai.creator.arithmetic import FxpParams
from elasticai.creator_plugins.quantized_grads.base_modules import BatchNorm2d
from elasticai.creator_plugins.quantized_grads.fixed_point import (
    QuantizeForwHTE,
    QuantizeParamToFixedPointHTE,
    QuantizeParamToFixedPointStochastic,
)


def test_batchnorm_fxp_init():
    conf = FxpParams(8, 3)

    BatchNorm2d(
        math_ops=QuantizeForwHTE(forward_conf=conf),
        weight_quantization=QuantizeParamToFixedPointHTE(conf),
        bias_quantization=QuantizeParamToFixedPointStochastic(conf),
        num_features=3,
    )
