from typing import Callable, List, Optional, Tuple

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer

__all__ = ["QuantizedFxpSGD", "quantized_sgd"]

from elasticai.creator.nn.quantized_grads.fixed_point._quantize_to_fixed_point import (
    quantize,
)
from elasticai.creator.nn.quantized_grads.fixed_point._two_complement_fixed_point_config import (
    FixedPointConfigV2,
)


class _RequiredParameter:
    """Singleton class representing a required parameter for an Optimizer."""

    def __repr__(self) -> str:
        return "<required parameter>"


required = _RequiredParameter()


class QuantizedFxpSGD(Optimizer):
    def __init__(
        self,
        params,
        fxp_conf: FixedPointConfigV2,
        save_quantization_error: bool,
        lr=required,
        momentum=0,
        weight_decay=0,
        *,
        maximize: bool = False,
        foreach: Optional[bool] = None,
        differentiable: bool = False,
        debug_print: bool = False,
    ):
        if lr is not required and lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        defaults = dict(
            lr=lr,
            fxp_config=fxp_conf,
            save_quantization_error=save_quantization_error,
            momentum=momentum,
            weight_decay=weight_decay,
            maximize=maximize,
            foreach=foreach,
            differentiable=differentiable,
            debug_print=debug_print,
        )
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("maximize", False)
            group.setdefault("foreach", None)
            group.setdefault("differentiable", False)

    def _init_group(
        self,
        group,
        params_with_grad,
        d_p_list,
        momentum_buffer_list,
        quantization_error_list,
    ):
        has_sparse_grad = False

        for p in group["params"]:
            if p.grad is not None:
                params_with_grad.append(p)
                d_p_list.append(p.grad)
                if p.grad.is_sparse:
                    has_sparse_grad = True

                state = self.state[p]
                if "momentum_buffer" not in state:
                    momentum_buffer_list.append(None)
                else:
                    momentum_buffer_list.append(state["momentum_buffer"])

                if "quantization_error" not in state:
                    quantization_error_list.append(None)
                else:
                    quantization_error_list.append(state["quantization_error"])
        return has_sparse_grad

    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            d_p_list = []
            momentum_buffer_list = []
            quantization_error_list = []

            self._init_group(
                group,
                params_with_grad,
                d_p_list,
                momentum_buffer_list,
                quantization_error_list,
            )

            quantized_sgd(
                params_with_grad,
                d_p_list,
                momentum_buffer_list,
                quantization_error_list,
                weight_decay=group["weight_decay"],
                momentum=group["momentum"],
                lr=group["lr"],
                fxp_config=group["fxp_config"],
                maximize=group["maximize"],
                save_quantization_error=group["save_quantization_error"],
                foreach=group["foreach"],
                debug_print=group["debug_print"],
            )

            # update momentum_buffers in state
            for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
                state = self.state[p]
                state["momentum_buffer"] = momentum_buffer

            # update quantization error in state
            for p, quantization_error in zip(params_with_grad, quantization_error_list):
                state = self.state[p]
                state["quantization_error"] = quantization_error

        return loss


def quantized_sgd(
    params: List[Tensor],
    d_p_list: List[Tensor],
    momentum_buffer_list: List[Optional[Tensor]],
    quantization_error_list: List[Optional[Tensor]],
    # kwonly args with defaults are not supported by functions compiled with torchscript issue #70627
    # setting this as kwarg for now as functional API is compiled by torch/distributed/optim
    foreach: Optional[bool] = None,
    *,
    weight_decay: float,
    momentum: float,
    lr: float,
    fxp_config: FixedPointConfigV2,
    save_quantization_error: bool,
    maximize: bool,
    debug_print: bool = False,
):
    r"""Functional API that performs SGD algorithm computation.

    See :class:`~torch.optim.SGD` for details.
    """

    func = _single_tensor_sgd
    func(
        params,
        d_p_list,
        momentum_buffer_list,
        quantization_error_list,
        weight_decay=weight_decay,
        momentum=momentum,
        lr=lr,
        fxp_config=fxp_config,
        save_quantization_error=save_quantization_error,
        maximize=maximize,
        debug_print=debug_print,
    )


def _single_tensor_sgd(
    params: List[Tensor],
    d_p_list: List[Tensor],
    momentum_buffer_list: List[Optional[Tensor]],
    quantization_error_list: List[Optional[Tensor]],
    *,
    weight_decay: float,
    momentum: float,
    lr: float,
    fxp_config: FixedPointConfigV2,
    save_quantization_error: bool,
    maximize: bool,
    debug_print: bool = False,
):
    for i, param in enumerate(params):
        with torch.no_grad():
            d_p = d_p_list[i] if not maximize else -d_p_list[i]
            if weight_decay != 0:
                raise NotImplementedError(f"Weight Decay is not properly implemented.")
                d_p = d_p.add(param, alpha=weight_decay)

            if momentum != 0:
                momentum_buf = momentum_buffer_list[i]

                if momentum_buf is None:
                    momentum_buf = quantize(torch.clone(d_p).detach(), fxp_config)
                    momentum_buffer_list[i] = momentum_buf
                else:
                    momentum_buf = quantize(
                        momentum_buf.mul_(momentum).add_(d_p, alpha=1), fxp_config
                    )
                d_p = momentum_buf

            if debug_print:
                print()
                print(f"OPTIMIZER {i}")
                print(f"Before update: {param=}")
                print(f"{d_p=}")
                print(f"{torch.unique(d_p)=}")
                print(f"{torch.mul(d_p, -lr)=}")

            if save_quantization_error:
                quant_buf = quantization_error_list[i]
                if debug_print:
                    print(f"{quant_buf=}")
                if quant_buf is None:
                    quant_buf = torch.zeros_like(d_p)
                delta = torch.add(torch.mul(d_p, -lr), quant_buf)
            else:
                delta = torch.mul(d_p, -lr)
            quantized_delta = quantize(delta, fxp_config)
            if debug_print:
                print(f"{quantized_delta=}")
            if save_quantization_error:
                quantization_error = delta - quantized_delta
                quantization_error_list[i] = quantization_error
                if debug_print:
                    print(f"{quantization_error=}")
            torch.add(param, quantized_delta, out=param)
            if debug_print:
                print(f"After update: {param=}")
