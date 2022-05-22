from typing import Tuple

from brevitas.export import BrevitasONNXManager
from torch import nn
from torch.nn import Module

from elasticai.creator.brevitas.translation_mapping import ConversionMapping


class BrevitasRepresentation:
    """
    This class provides an interface for the user translate from qtorch to brevitas.
    expected starting point:
    brevitas_representation = BrevitasRepresentation.from_pytorch(model)
    """

    def __init__(self, original_model: Module, translated_model: Module) -> None:
        self._original_model = original_model
        self._translated_model = translated_model

    @staticmethod
    def _translate_model(model: Module) -> Module:
        """
        translates sequential qtorch model to a sequential brevitas model
        Args:
            model (Module): qtorch model
        Returns:
            brevitas model
        """
        mapping = ConversionMapping()
        translated_layers = []

        for layer in model:
            conversion_function = mapping.get_conversion_function(layer)
            translated_layer = conversion_function(layer)
            translated_layers.append(translated_layer)

        return nn.Sequential(*translated_layers)

    @staticmethod
    def from_pytorch(model: Module):
        """
        translates sequential qtorch model to a sequential brevitas model and returns the BrevitasRepresentation
        Args:
            model (Module): qtorch model
        Returns:
            Brevitas Representation
        """
        translated_model = BrevitasRepresentation._translate_model(model)
        return BrevitasRepresentation(model, translated_model)

    @property
    def original_model(self) -> Module:
        """
        getter for original qtorch model
        Returns:
            original qtorch model
        """
        return self._original_model

    @property
    def translated_model(self) -> Module:
        """
        getter for translated brevitas model
        Returns:
            translated brevitas model
        """
        return self._translated_model

    def save_to_onnx(self, input_shape: Tuple, path: str) -> None:
        """
        saves the translated brevitas model in an onnx file with the brevitas specific conversion
        Args:
            input_shape (Tuple): input shape
            path (str): path to onnx file
        """
        # we use brevitas onnx manager because the standard onnx manager doesn't work for our model
        # (https://github.com/Xilinx/brevitas/issues/365)
        BrevitasONNXManager.export(
            module=self.translated_model, input_shape=input_shape, export_path=path
        )
