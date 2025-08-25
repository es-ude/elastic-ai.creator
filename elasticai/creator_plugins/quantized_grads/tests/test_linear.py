import torch

from elasticai.creator.arithmetic import FxpParams
from elasticai.creator_plugins.quantized_grads.base_modules import Linear
from elasticai.creator_plugins.quantized_grads.fixed_point import (
    QuantizeForwHTE,
    QuantizeParamToFixedPointHTE,
    QuantizeParamToFixedPointStochastic,
)


def test_conv1d_fxp_init():
    conf = FxpParams(total_bits=8, frac_bits=3)

    l = Linear(
        QuantizeForwHTE(conf),
        2,
        3,
        QuantizeParamToFixedPointHTE(conf),
        True,
        bias_quantization=QuantizeParamToFixedPointStochastic(conf),
    )

    # print(l.)
    x = torch.randn(1, 1, 2)
    l(x)
