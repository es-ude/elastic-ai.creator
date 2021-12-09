from typing import Dict

from torch.nn import Module
from elasticai.creator.layers import Binarize, Ternarize
import elasticai.creator.translator.brevitas.brevitas_quantizers as bquant


def set_quantizers(layer: Module, args: Dict) -> None:
    """
    Function to do the mapping between brevitas and Qtorch layer quantizers.
    The function adds the quantizer to the arguments of the layer.
    Args:
        layer (Layer): current layer
        args (Dict): The layer args
    """
    if isinstance(layer.quantizer, Binarize):
        args["weight_quant"] = bquant.BinaryWeights
        if args["bias"]:
            args["bias_quant"] = bquant.BinaryBias
    elif isinstance(layer.quantizer, Ternarize):
        args["weight_quant"] = bquant.TernaryWeights
        if args["bias"]:
            args["bias_quant"] = bquant.TernaryBias
