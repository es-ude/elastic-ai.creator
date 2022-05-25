import torch
from torch.nn import Conv1d
from torch.nn.utils.parametrize import register_parametrization

from elasticai.creator.qat.layers import Binarize

layer = Conv1d(in_channels=2, out_channels=3, kernel_size=(1,), bias=False)

register_parametrization(layer, "weight", Binarize())

print(layer)
x = torch.tensor([[[1], [1]]], dtype=torch.float32)
y = layer(x)
print(y)
