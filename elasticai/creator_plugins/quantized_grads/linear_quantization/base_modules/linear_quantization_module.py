import torch
from torch._C._te import Tensor

from elasticai.creator_plugins.quantized_grads.linear_quantization.module_quantization import QuantizationModule


class LinearQuantizationLayer(torch.nn.Module):
    ...
