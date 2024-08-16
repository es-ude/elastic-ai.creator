import logging

import torch

from elasticai.creator.nn.integer.quant_utils.QuantizedTensorValidator import (
    QuantizedTensorValidator,
)


def substraction(
    a: torch.IntTensor, b: torch.IntTensor, c_quant_bits: int
) -> torch.IntTensor:
    logger = logging.getLogger(__name__)

    QuantizedTensorValidator.check_dtype(a, "a", torch.int32, logger)
    QuantizedTensorValidator.check_dtype(b, "b", torch.int32, logger)

    c = a - b

    QuantizedTensorValidator.check_dtype(c, "c", torch.int32, logger)
    QuantizedTensorValidator.check_drange(
        c, "c", -(2 ** (c_quant_bits - 1)), (2 ** (c_quant_bits - 1)) - 1, logger
    )
    return c
