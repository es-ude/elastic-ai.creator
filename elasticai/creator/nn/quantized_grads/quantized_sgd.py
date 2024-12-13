from typing import Callable

import torch.nn
from torch import Tensor
from torch.nn import Sequential

from elasticai.creator.nn.quantized_grads.quantized_parameters import (
    QuantizationParameter,
    QuantizationSchemeByName,
    QuantizedParameters,
)


class QuantizedSGD:
    def __init__(self, model, lr: float = 0.01, momentum: float = 0) -> None:
        self.model: Sequential = model
        self.lr = lr
        self.momentum = momentum

    def step(self):
        qparams = self.get_all_params_over_sequential(self.model)
        for i, qparam in enumerate(qparams):
            single_tensor_sgd(
                p=qparam.parameter,
                d_p=qparam.gradient,
                momentum_buffer=torch.empty_like(qparam.parameter),
                lr=self.lr,
                momentum=self.momentum,
                weight_decay=0,
                quantization=qparam.quantization,
            )

    def get_all_params_over_sequential(
        self, sequential: Sequential
    ) -> list[QuantizationParameter]:
        all_qparams = []
        for layer in sequential:
            if isinstance(layer, Sequential):
                all_qparams.extend(self.get_all_params_over_sequential(layer))
            elif isinstance(layer, QuantizedParameters):
                for i, param in enumerate(layer.qparams):
                    qparam_name = param.name
                    qparam_quantization = param.quantization
                    qparam_parameter = layer._parameters[qparam_name].data
                    qparam_gradient = layer._parameters[qparam_name].grad
                    qparam = QuantizationParameter(
                        qparam_name,
                        qparam_quantization,
                        qparam_parameter,
                        qparam_gradient,
                    )
                    all_qparams.append(qparam)
        return all_qparams


def single_tensor_sgd(
    p: Tensor,
    d_p: Tensor,
    momentum_buffer: Tensor,
    lr: float,
    momentum: float,
    weight_decay: float,
    quantization: Callable[[Tensor], Tensor],
) -> None:
    """
    First Version of SGD. Momentum+buffer needs to be implemented soon. Weight decay will likely be implemented too.
    """
    with torch.no_grad():
        p.add_(-d_p, alpha=lr)
        quantization(p)
