from dataclasses import dataclass

import torch


# todo: change name of class
@dataclass
class FixedPointConfigV2:
    total_bits: int
    frac_bits: int
    stochastic_rounding: bool

    @property
    def conf_is_valid(self) -> bool:
        valid = True
        if self.total_bits <= 0:
            valid = False
        if self.frac_bits + 1 > self.total_bits:
            valid = False
        return valid

    @property
    def minimum_as_rational(self):
        return -(2 ** (self.total_bits - self.frac_bits - 1))

    @property
    def maximum_as_rational(self):
        return -self.minimum_as_rational - 1 / (2**self.frac_bits)

    # Todo: Is this needed? Otherwise remove it
    # @property
    # def device(self):
    #    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    @property
    def minimum_as_rational_tensor(self):
        return torch.Tensor([self.minimum_as_rational])

    @property
    def maximum_as_rational_tensor(self):
        return torch.Tensor([self.maximum_as_rational])

    def quantize(self, number: torch.Tensor) -> torch.Tensor:
        if self.conf_is_valid:
            return self._round(self.clamp(number))
        return number

    def _round(self, number: torch.Tensor) -> torch.Tensor:
        if self.conf_is_valid:
            if self.stochastic_rounding:
                noise = (torch.rand_like(number) - 0.5) / (2**self.frac_bits)
                return round_to_fixed_point(number + noise, self.frac_bits)
            else:
                return round_to_fixed_point(number, self.frac_bits)
        else:
            return number

    def clamp(self, number: torch.Tensor) -> torch.Tensor:
        if self.conf_is_valid:
            return torch.clamp(
                number,
                self.minimum_as_rational_tensor,
                self.maximum_as_rational_tensor,
            )
        else:
            return number


def round_to_fixed_point(number: torch.Tensor, frac_bits: int) -> torch.Tensor:
    return torch.round(number * (2**frac_bits)) / 2**frac_bits
