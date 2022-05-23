import brevitas.nn as bnn

import elasticai.creator.brevitas.brevitas_quantizers as bquant
from elasticai.creator.layers import Binarize, Ternarize


def translate_binarize_layer(layer: Binarize) -> bnn.QuantIdentity:
    """
    translates a qtorch binarize layer into a brevitas binarize layer
    Args:
        layer (Binarize): qtorch layer
    Returns:
        brevitas layer
    """
    return bnn.QuantIdentity(act_quant=bquant.BinaryActivation)


def translate_ternarize_layer(layer: Ternarize) -> bnn.QuantIdentity:
    """
    translates a qtorch ternarize layer into a brevitas ternarize layer
    Args:
        layer (Ternarize): qtorch layer
    Returns:
        brevitas layer
    """
    return bnn.QuantIdentity(act_quant=bquant.TernaryActivation)
