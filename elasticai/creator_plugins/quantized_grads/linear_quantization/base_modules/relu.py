from torch import Tensor
from torch.nn import ReLU as TorchReLU

from elasticai.creator_plugins.quantized_grads.linear_quantization.base_modules.linear_quantization_module import \
    LinearQuantizationLayer
from elasticai.creator_plugins.quantized_grads.linear_quantization.module_quantization import QuantizationModule


class ReLU(LinearQuantizationLayer, TorchReLU):
    ...
