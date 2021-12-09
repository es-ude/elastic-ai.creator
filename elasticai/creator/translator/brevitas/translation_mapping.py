from typing import Callable

from elasticai.creator.translator.brevitas.translation_functions import translate_conv1d, translate_conv2d, \
    translate_linear_layer, translate_binarize_layer, translate_ternarize_layer, translate_layer
from torch.nn import  Module


class ConversionMapping:
    """
    Class to store all currently possible qtorch->brevitas mappings.
    """
    def __init__(self):
        self._mapping = {
            "Binarize": translate_binarize_layer,
            "Ternarize": translate_ternarize_layer,
            "QLinear": translate_linear_layer,
            "QConv1d": translate_conv1d,
            "QConv2d": translate_conv2d,
            "MaxPool1d": translate_layer,
            "BatchNorm1d": translate_layer,
            "Flatten": translate_layer,
            "Sigmoid": translate_layer

        }

    def get_conversion_function(self, layer: Module) -> Callable[[Module], Module]:
        """
        return conversion function for passed layer
        Args:
            layer (Layer): layer which should be converted
        Returns:
            conversion function for layer
        """
        layer_name = layer.__class__.__name__
        try:
            return self._mapping[layer_name]
        except KeyError:
            raise NotImplementedError(f"For the layer {layer_name} no mapping to a conversion function is implemented.")
