from typing import Optional, Tuple

from torch.nn import  Module
from torch import  Tensor
from torch.onnx import register_custom_op_symbolic
import torch
from elasticai.creator.precomputation import Precalculation_function


class onnx_export_manager:
    """
    Adds function specification to allow to export models
    """
    @classmethod
    def export_onnx(cls,module: Module,
            input_shape: Tuple[int, ...],
            export_path: str = None,
             **kwargs):
        """
        Takes a Pytorch model and exports it to ONNX
        Args:
            module: A pytorch module
            input_shape: An example input for the ONNX.
            export_path: A pathlike string where the model will be saved
        """

        register_custom_op_symbolic("custom_ops::Precomputation", Precalculation_function, 9)
        torch.onnx.export(module, torch.ones(input_shape), export_path, **kwargs)
        pass