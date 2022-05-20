import brevitas.nn as bnn
from torch.nn import Module

from elasticai.creator.brevitas.translation_functions.translation_function_tools import (
    set_quantizers,
)
from elasticai.creator.layers import QLinear


def translate_linear_layer(layer: QLinear) -> bnn.QuantLinear:
    """
    translates a qtorch linear layer into a brevitas linear layer
    the arguments convertion of the quantizers is done in an extra file because it is also used by the convolutional layers
    Args:
        layer (Linear): qtorch layer
    Returns:
        brevitas layer
    """

    args = {
        "in_features": layer.in_features,
        "out_features": layer.out_features,
        "bias": layer.bias is not None,
    }

    set_quantizers(layer, args)

    return bnn.QuantLinear(**args)
