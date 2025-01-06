from typing import cast

import torch

from elasticai.creator.nn.quantized_grads._math_operations import MathOperations
from elasticai.creator.nn.quantized_grads.fixed_point import FixedPointConfigV2
from elasticai.creator.nn.quantized_grads.fixed_point._round_to_fixed_point_autograd import (
    QuantizeForwBackwHTE,
    QuantizeForwBackwStochastic,
    QuantizeForwHTE,
    QuantizeForwStochastic,
)


class MathOperationsForwStoch(MathOperations):
    def __init__(self, forward: FixedPointConfigV2) -> None:
        self.forward = forward

    def quantize(self, a: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, QuantizeForwStochastic.apply(a, self.forward))


class MathOperationsForwHTE(MathOperations):
    def __init__(self, forward: FixedPointConfigV2) -> None:
        self.forward = forward

    def quantize(self, a: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, QuantizeForwHTE.apply(a, self.forward))


class MathOperationsForwBackwStoch(MathOperations):
    def __init__(
        self, forward: FixedPointConfigV2, backward: FixedPointConfigV2
    ) -> None:
        self.forward = forward
        self.backward = backward

    def quantize(self, a: torch.Tensor) -> torch.Tensor:
        return cast(
            torch.Tensor,
            QuantizeForwBackwStochastic.apply(a, self.forward, self.backward),
        )


class MathOperationsForwBackwHTE(MathOperations):
    def __init__(
        self, forward: FixedPointConfigV2, backward: FixedPointConfigV2
    ) -> None:
        self.forward = forward
        self.backward = backward

    def quantize(self, a: torch.Tensor) -> torch.Tensor:
        return cast(
            torch.Tensor, QuantizeForwBackwHTE.apply(a, self.forward, self.backward)
        )
