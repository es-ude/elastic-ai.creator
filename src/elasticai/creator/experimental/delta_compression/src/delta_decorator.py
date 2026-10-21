import logging
from typing import Any

import torch
from torch import Tensor
from torch.nn import Module

from elasticai.creator.base_modules.conv1d import MathOperations as Conv1dMathOperations
from elasticai.creator.base_modules.linear import MathOperations as LinearMathOperations
from elasticai.creator.base_modules.lstm_cell import (
    MathOperations as LSTMMathOperations,
)
from elasticai.creator.experimental.delta_compression import DeltaCompression

logger = logging.getLogger(__name__)

type MathOperations = LinearMathOperations | Conv1dMathOperations | LSTMMathOperations


def _format_tensor_full(name: str, tensor: Tensor) -> str:
    """Format a tensor for logging, printing every value without truncation."""
    values = tensor.detach().cpu().tolist()
    return f"  {name}: shape={tuple(tensor.shape)} dtype={tensor.dtype} values={values}"


class _DeltaCompressedParamSTE(torch.autograd.Function):
    """STE for delta compressed parameters: quantize -> compress -> inflate.

    Gradients pass through unchanged (Straight-Through Estimator).
    """

    @staticmethod
    def forward(
        ctx, param: Tensor, ops: MathOperations, dc: DeltaCompression
    ) -> Tensor:
        if not hasattr(ops, "config"):
            raise TypeError(
                "delta_compressed_* decorators require 'config' attribute. "
            )
        if not hasattr(ops.config, "cut_as_integer"):
            raise TypeError(
                "delta_compressed_* decorators require 'cut_as_integer' method. "
            )
        if not hasattr(ops.config, "as_rational"):
            raise TypeError(
                "delta_compressed_* decorators require 'as_rational' method. "
            )

        # FIXME: Warning for non resovolable attribute `ops.config` requires new MathOpertaions parent class!

        quant_tensor = ops.quantize(param)
        int_tensor = ops.config.cut_as_integer(quant_tensor).to(torch.int32)
        compressed_tensor = dc.compress(int_tensor, in_place=False)
        inflated_tensor = dc.inflate(compressed_tensor, in_place=False)
        reconstructed = ops.config.as_rational(inflated_tensor)

        logger.info("Applied delta compression (STE) on parameter!")
        if logger.isEnabledFor(logging.DEBUG):
            tensors = {
                "param": param,
                "quant_tensor": quant_tensor,
                "int_tensor": int_tensor,
                "compressed_tensor": compressed_tensor,
                "inflated_tensor": inflated_tensor,
                "reconstructed": reconstructed,
            }
            logger.debug(
                "DeltaCompressedParamSTE:\n%s",
                "\n".join(
                    _format_tensor_full(name, tensor)
                    for name, tensor in tensors.items()
                ),
            )
            logger.debug(
                "reconstructed matches original (allclose)=%s",
                torch.allclose(param, reconstructed),
            )

        return reconstructed

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple[Tensor, None, None]:  # ty:ignore[invalid-method-override]
        return grad_output, None, None


class _DeltaCompressedParamDescriptor:
    """Descriptor that applies delta compression STE transformation on parameter read."""

    def __init__(self, dc: DeltaCompression, private_name: str) -> None:
        self.dc = dc
        self.private_name = private_name

    def __get__(self, obj: Module, objtype: type | None = None) -> Tensor:
        if obj is None:
            return self
        original = getattr(obj, self.private_name)
        logger.debug(
            "Accessing delta compressed parameter '%s' of module '%s'",
            self.private_name,
            obj.__class__.__name__,
        )
        return _DeltaCompressedParamSTE.apply(original, obj._operations, self.dc)

    def __set__(self, obj: Module, value: Tensor) -> None:
        setattr(obj, self.private_name, value)


def _create_param_decorator(param_name: str):
    """Factory for creating parameter-specific decorators with elastic layer validation."""

    def decorator(dc: DeltaCompression):
        def wrapper(cls: type) -> type:
            original_init = cls.__init__

            def new_init(self: Any, *args: Any, **kwargs: Any) -> None:
                original_init(self, *args, **kwargs)

                if not hasattr(self, "_operations"):
                    raise TypeError(
                        f"delta_compressed_* decorators require '_operations' attribute. "
                        f"Use only with elasticai layers. Layer: {cls.__name__}"
                    )

                if hasattr(self, param_name):
                    param = getattr(self, param_name)
                    if isinstance(param, torch.nn.Parameter):
                        private_name = f"_original_{param_name}"
                        setattr(self, private_name, param)
                        setattr(
                            cls,
                            param_name,
                            _DeltaCompressedParamDescriptor(dc, private_name),
                        )

            setattr(cls, "__init__", new_init)
            return cls

        return wrapper

    return decorator


# Public decorators for weights and bias
delta_compressed_weights = _create_param_decorator("weight")
delta_compressed_bias = _create_param_decorator("bias")
