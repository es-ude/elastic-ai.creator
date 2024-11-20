from typing import Protocol

from torch.nn import MaxPool2d as _MaxPool2d

from elasticai.creator.base_modules.math_operations import Quantize


class MathOperations(Quantize, Protocol):
    ...


class MaxPool2d(_MaxPool2d):
    def __init__(
        self,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] | None = None,
        padding: int | tuple[int, int] = 0,
        dilation: int | tuple[int, int] = 1,
        return_indices: bool = False,
        ceil_mode: bool = False,
    ) -> None:
        super().__init__(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            return_indices=return_indices,
            ceil_mode=ceil_mode,
        )
