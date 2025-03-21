from typing import Callable

import torch
from torch import Tensor, optim
from torch.nn import Module

from elasticai.creator_plugins.quantized_grads.base_modules import parametrized_modules


class _QOptim(optim.Optimizer):
    def __init__(
        self,
        model,
        *args,
        buffer_quantizations: dict[str, Callable[[Tensor], Tensor]] = None,
        **kwargs,
    ): ...


def get_quantized_optimizer(optimizer: type[optim.Optimizer]) -> type[_QOptim]:
    class QuantizedOptim(optimizer):
        def __init__(
            self,
            model: Module,
            *args,
            buffer_quantizations: dict[str, Module] = None,
            **kwargs,
        ):
            super().__init__(*args, **kwargs)
            self._model = model
            self.register_step_post_hook(_ensure_tensors_quantization_after_update)
            if buffer_quantizations is None:
                self.buffer_quantization = {}
            else:
                self.buffer_quantization: dict[str, Callable[[Tensor], Tensor]] = (
                    buffer_quantizations
                )
                self.register_step_post_hook(_quantization_for_buffers_in_optimizer)

        def step(self, closure=None):
            super().step(closure)

    def _ensure_tensors_quantization_after_update(
        optimizer: QuantizedOptim, *args, **kwargs
    ):
        with torch.no_grad():
            for m in optimizer._model.modules():
                if m.__class__.__name__ in parametrized_modules:
                    for p in m.parametrizations:
                        m_parametrization_param = m.parametrizations[p]
                        m_param = getattr(m, p)
                        m_param.data = m_parametrization_param[0].right_inverse(m_param)

    def _quantization_for_buffers_in_optimizer(
        optimizer: QuantizedOptim, *args, **kwargs
    ):
        for group in optimizer.param_groups:
            for p in group["params"]:
                state = optimizer.state[p]
                for key in state.keys():
                    if key in optimizer.buffer_quantization.keys():
                        if (
                            state[key].device
                            != optimizer.buffer_quantization[key].device
                        ):
                            optimizer.buffer_quantization[key].to(state[key].device)
                        state[key] = optimizer.buffer_quantization[key](state[key])

    return QuantizedOptim
