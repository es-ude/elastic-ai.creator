from dataclasses import dataclass

import torch


@dataclass
class FixedPointConfigV2:
    def __init__(self, total_bits: int, frac_bits: int, stochastic_rounding: bool):
        self.total_bits: int = total_bits
        self.frac_bits: int = frac_bits
        self.stochastic_rounding: bool = stochastic_rounding
        if self.total_bits <= 0:
            raise Exception(
                f"total bits need to be greater than 0 for {self.__class__.__name__}"
            )
        if self.frac_bits + 1 > self.total_bits:
            raise Exception(
                f"total bits-1 needs to be greater than frac bits for {self.__class__.__name__}"
            )

    @property
    def minimum_as_rational(self):
        return -(2 ** (self.total_bits - self.frac_bits - 1))

    @property
    def maximum_as_rational(self):
        return -self.minimum_as_rational - 1 / (2**self.frac_bits)

    @property
    def minimum_as_rational_tensor(self):
        return torch.Tensor([self.minimum_as_rational])

    @property
    def maximum_as_rational_tensor(self):
        return torch.Tensor([self.maximum_as_rational])
