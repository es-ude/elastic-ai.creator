from typing import Callable

import torch.nn
from torch import Tensor
from torch.nn import Sequential

from elasticai.creator.nn.quantized_grads.quantized_parameters import (
    QuantizationParameter,
    QuantizedParameters,
)


class QuantizedSGD:
    def __init__(
        self, model, lr: float = 0.01, momentum: float = 0, weight_decay: float = 0
    ) -> None:
        self.model: Sequential = model
        self.lr = lr
        self.momentum = momentum
        self.momentum_buffers: list = list()
        self._init_momentum_buffers()
        self.weight_decay = weight_decay

    def _init_momentum_buffers(self) -> None:
        qparams = self._get_all_params_from_sequential(self.model)
        for i, qparam in enumerate(qparams):
            self.momentum_buffers.append(torch.zeros_like(qparam.parameter))

    def step(self):
        qparams = self._get_all_params_from_sequential(self.model)
        for i, qparam in enumerate(qparams):
            single_tensor_sgd(
                p=qparam.parameter,
                d_p=qparam.gradient,
                momentum_buffer=self.momentum_buffers[i],
                lr=self.lr,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
                tensor_quantization_=qparam.quantization,
            )

    def zero_grad(self):
        self.model.zero_grad()

    def _get_all_params_from_sequential(
        self, sequential: Sequential
    ) -> list[QuantizationParameter]:
        all_qparams = []
        for layer in sequential:
            if isinstance(layer, Sequential):
                all_qparams.extend(self._get_all_params_from_sequential(layer))
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

    def state_dict(self) -> dict:
        return {
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "momentum": self.momentum,
            "momentum_buffers": self.momentum_buffers,
        }

    def load_state_dict(self, state_dict: dict):
        self.lr = state_dict["lr"]
        self.weight_decay = state_dict["weight_decay"]
        self.momentum = state_dict["momentum"]
        self.momentum_buffers = state_dict["momentum_buffers"]


def single_tensor_sgd(
    p: Tensor,
    d_p: Tensor,
    momentum_buffer: Tensor,
    lr: float,
    momentum: float,
    weight_decay: float,
    tensor_quantization_: Callable[[Tensor], None],
) -> None:
    """
    IMPORTANT: tensor_quantization_ must be inplace operation.
    """
    with torch.no_grad():
        if weight_decay != 0.0:
            d_p = d_p.add(p, alpha=weight_decay)
        if momentum != 0.0:
            momentum_buffer.mul_(momentum).add_(d_p)
            d_p = momentum_buffer
            tensor_quantization_(momentum_buffer)
        p.add_(-d_p, alpha=lr)
        tensor_quantization_(p)
