from typing import Callable

import torch
from torch.nn.parameter import Parameter

from elasticai.creator.vhdl.number_representations import FixedPointFactory

OperationType = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def _default_matmul_op(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.matmul(a, b)


def _default_add_op(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.add(a, b)


class Linear(torch.nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        matmul_op: OperationType = _default_matmul_op,
        add_op: OperationType = _default_add_op,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)
        self._matmul_op = matmul_op
        self._add_op = add_op

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.bias is None:
            return self._matmul_op(self.weight, x)
        return self._add_op(self._matmul_op(self.weight, x), self.bias)


class FixedPointLinear(Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        fixed_point_factory: FixedPointFactory,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            matmul_op=self._fp_matmul_op,
            add_op=_default_add_op,
            device=device,
            dtype=dtype,
        )
        self._to_fp = fixed_point_factory

    def _fp_matmul_op(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return (torch.matmul(a, b) / self._to_fp(1).to_signed_int()).floor()
