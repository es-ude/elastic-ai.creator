from typing import Union

import brevitas.nn as bnn

from elasticai.creator.brevitas.translation_functions.translation_function_tools import (
    set_quantizers,
)
from elasticai.creator.layers import QConv1d, QConv2d


def translate_conv(layer: Union[QConv1d, QConv2d]) -> dict:
    """
    translates the arguments of a qtorch convolutional layer into the arguments of a brevitas convolutional layer
    the padding mode of the qtorch layer is translated to the padding type of the brevitas layer, brevitas does not support all padding modes
    the arguments conversion of the quantizers is done in an extra file because it is also used by the linear layer
    Args:
        layer (Conv): current convolutional layer
    Returns:
        a dictionary with all arguments for the brevitas convolutional layer
    """
    args = {
        "in_channels": layer.in_channels,
        "out_channels": layer.out_channels,
        "kernel_size": layer.kernel_size,
        "stride": layer.stride,
        "padding": layer.padding,
        "padding_type": "standard",
        "dilation": layer.dilation,
        "groups": layer.groups,
        "bias": layer.bias is not None,
    }

    if layer.padding_mode == "zeros":
        if isinstance(layer.padding, str):
            if layer.padding == "same":
                args["padding_type"] = "same"
                args["padding"] = 0
            else:
                raise NotImplementedError(f"Padding {layer.padding} not implemented.")
    else:
        raise NotImplementedError(f"Padding mode {layer.padding} not implemented.")

    set_quantizers(layer, args)

    return args


def translate_conv1d(layer: QConv1d) -> bnn.QuantConv1d:
    """
    translates a 1d qtorch convolutional layer into a 1d brevitas convolutional layer
    Args:
        layer (Conv): qtorch layer
    Returns:
        brevitas layer
    """
    args = translate_conv(layer=layer)
    return bnn.QuantConv1d(**args)


def translate_conv2d(layer: QConv2d) -> bnn.QuantConv2d:
    """
    translates a 2d qtorch convolutional layer into a 2d brevitas convolutional layer
    Args:
        layer (Conv): qtorch layer
    Returns:
        brevitas layer
    """
    args = translate_conv(layer=layer)
    return bnn.QuantConv2d(**args)
