from typing import Optional, Tuple

from torch.nn import  Module
from torch import  Tensor
from torch.onnx import register_custom_op_symbolic
import torch
from elasticai.creator.precomputation import Precalculation_function


class onnx_export_manager:
    @classmethod
    def export_onnx(cls,module: Module,
            input_shape: Optional[Tuple[int, ...]] = None,
            export_path: Optional[str] = None,
            input_t: Optional[Tensor] = None, **kwargs):

        register_custom_op_symbolic("custom_ops::Precomputation", Precalculation_function, 9)
        torch.onnx.export(module, torch.ones(input_shape), export_path, **kwargs)
        pass