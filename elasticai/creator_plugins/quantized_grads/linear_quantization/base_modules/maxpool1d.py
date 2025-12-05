from torch.nn import MaxPool1d as TorchMaxPool1d
from torch import Tensor

from elasticai.creator_plugins.quantized_grads.linear_quantization.base_modules.linear_quantization_module import \
    LinearQuantizationLayer
from elasticai.creator_plugins.quantized_grads.linear_quantization.module_quantization import QuantizationModule


class MaxPool1d(LinearQuantizationLayer, TorchMaxPool1d):
    ...
