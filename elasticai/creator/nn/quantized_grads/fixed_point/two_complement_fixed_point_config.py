from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class FixedPointConfigV2:
    total_bits: int
    frac_bits: int

    def __post_init__(self):
        """
        This python API call for dataclass is executed after init.
        This way we check if the FixedPointConfigV2 is valid while keeping the config immutable.
        """
        if self.total_bits <= 0:
            raise Exception(
                f"total bits need to be > 0 for {self.__class__.__name__}. "
                f"You have set {self.total_bits=}."
            )
        if self.frac_bits + 1 > self.total_bits:
            raise Exception(
                f"total bits-1 needs to be > frac bits for {self.__class__.__name__}."
                f"You have set {self.total_bits=} and {self.frac_bits=}."
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
