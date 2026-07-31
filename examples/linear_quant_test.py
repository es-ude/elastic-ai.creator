from typing import Protocol

import torch
from torch.nn import AdaptiveAvgPool1d as _AdaptiveAvgPool1d
from torch.nn import Sequential

from elasticai.creator.base_modules.conv1d import Conv1d
from elasticai.creator.base_modules.linear import Linear
from elasticai.creator.base_modules.math_operations import Add, Mul, Quantize
from elasticai.creator.nn.linear import (
    LinearQuantArithmetic,
    LinearQuantMathOps,
)
from elasticai.creator.nn.linear import (
    SymetricLinearQuantParams as LinearQuantParams,
)

##################################################
## AveragePooling1D


class AveragePooling1dMathOps(Quantize, Mul, Add, Protocol): ...


class AdaptiveAveragePooling1d(_AdaptiveAvgPool1d):
    def __init__(
        self, output_size: int | tuple[int], operations: AveragePooling1dMathOps
    ) -> None:
        super().__init__(output_size)
        self._operations = operations

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        pooled = super().forward(input)
        return self._operations.quantize(pooled)


##################################################
## EXAMPLE


def main():
    # setup linear quantization methods for QAT
    params = LinearQuantParams(float_range=(-1.0, 1.0), total_bits=8, signed=True)
    arithmetic = LinearQuantArithmetic(params=params)
    math_ops = LinearQuantMathOps(config=arithmetic)

    # setup neural network layers
    model = Sequential(
        Conv1d(
            in_channels=3,
            out_channels=6,
            kernel_size=3,
            stride=1,
            padding=0,
            operations=math_ops,
            bias=False,
        ),
        Conv1d(
            in_channels=6,
            out_channels=12,
            kernel_size=3,
            stride=1,
            padding=0,
            operations=math_ops,
            bias=False,
        ),
        AdaptiveAveragePooling1d(output_size=1, operations=math_ops),
        Linear(in_features=12, out_features=5, operations=math_ops, bias=False),
    )

    print(model)
    print(model.state_dict())


if __name__ == "__main__":
    main()
