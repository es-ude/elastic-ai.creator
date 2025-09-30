from dataclasses import dataclass

import torch
from torch import Tensor

@dataclass(frozen=True)
class LinearQuantizationConfig:
    num_bits: int

    @property
    def min_value(self) -> Tensor:
        return Tensor([0.])

    @property
    def max_value(self)-> Tensor:
        return Tensor([2**self.num_bits-1])


@dataclass(frozen=True)
class IntQuantizationConfig(LinearQuantizationConfig):
    @property
    def min_value(self) -> Tensor:
        return Tensor([-2**(self.num_bits-1)])

    @property
    def max_value(self)-> Tensor:
        return Tensor([2**(self.num_bits-1)-1])
