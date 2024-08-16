import logging

import torch

from elasticai.creator.nn.integer.quant_utils.QuantizedTensorValidator import (
    QuantizedTensorValidator,
)


def subtract(
    a: torch.IntTensor, b: torch.IntTensor, c_quant_bits: int
) -> torch.IntTensor:
    logger = logging.getLogger(__name__)
    c = a - b
    QuantizedTensorValidator.check_drange(
        c, "c", -(2 ** (c_quant_bits - 1)), (2 ** (c_quant_bits - 1)) - 1, logger
    )
    return c
