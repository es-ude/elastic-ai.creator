import torch

from elasticai.creator.vhdl.translator.abstract.layers import Linear1dModule


def build_linear_1d(linear: torch.nn.Linear) -> Linear1dModule:
    def to_list(tensor: torch.Tensor) -> list:
        return tensor.detach().numpy().tolist()

    return Linear1dModule(weight=to_list(linear.weight), bias=to_list(linear.bias))
