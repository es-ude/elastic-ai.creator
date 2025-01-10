import torch

from elasticai.creator.nn.quantized_grads.base_modules import BatchNorm2d
from elasticai.creator.nn.quantized_grads.fixed_point import (
    FixedPointConfigV2,
    MathOperationsForwHTE,
    quantize_to_fxp_stochastic_,
)


def test_batchnorm_fxp_init():
    conf = FixedPointConfigV2(8, 3)

    def quantize_weight(tensor: torch.Tensor) -> None:
        quantize_to_fxp_stochastic_(tensor, conf)

    def quantize_bias(tensor: torch.Tensor) -> None:
        quantize_to_fxp_stochastic_(tensor, conf)

    ops = MathOperationsForwHTE(conf)
    l = BatchNorm2d(
        operations=ops,
        weight_quantization=quantize_weight,
        bias_quantization=quantize_bias,
        num_features=3,
    )

    for qparam in l.qparams:
        assert qparam.name in ["weight", "bias"]
    if qparam.name == "weight":
        assert qparam.quantization == quantize_weight
    elif qparam.name == "bias":
        assert qparam.quantization == quantize_bias
    else:
        assert False