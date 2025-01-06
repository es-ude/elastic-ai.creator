import torch.nn

from elasticai.creator.nn.binary.binary_quantization_function import (
    Binarize as _BinarizeFn,
)


class Binarize(torch.nn.Module):
    def forward(self, X):
        return _BinarizeFn.apply(X)
