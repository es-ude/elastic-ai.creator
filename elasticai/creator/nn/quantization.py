from collections.abc import Callable
from dataclasses import dataclass

import torch

from elasticai.creator.nn.arithmetics import Arithmetics

QuantType = Callable[[torch.Tensor], torch.Tensor]
OperationType = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass
class FakeQuant:
    quant: QuantType
    dequant: QuantType
    arithmetics: Arithmetics

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.dequant(self.quant(self.arithmetics.clamp(x)))
