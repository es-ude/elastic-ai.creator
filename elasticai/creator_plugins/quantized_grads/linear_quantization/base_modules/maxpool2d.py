from torch.nn import MaxPool2d as TorchMaxPool2d
from torch import Tensor

from elasticai.creator_plugins.quantized_grads.linear_quantization.base_modules.linear_quantization_module import \
    LinearQuantizationLayer
from elasticai.creator_plugins.quantized_grads.linear_quantization.module_quantization import QuantizationModule


class MaxPool2d(LinearQuantizationLayer, TorchMaxPool2d):
    ...
