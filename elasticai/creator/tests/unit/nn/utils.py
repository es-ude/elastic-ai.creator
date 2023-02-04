import torch

from elasticai.creator.vhdl.number_representations import (
    FixedPoint,
    float_values_to_fixed_point,
)


def to_list(x: torch.Tensor) -> list:
    return x.detach().numpy().tolist()


def to_fixed_point(values: list[float], total_bits: int, frac_bits: int) -> list[int]:
    def to_fp(value: FixedPoint) -> int:
        return value.to_signed_int()

    return list(map(to_fp, float_values_to_fixed_point(values, total_bits, frac_bits)))


def from_fixed_point(values: list[int], total_bits: int, frac_bits: int) -> list[float]:
    def to_float(value: int) -> float:
        return float(FixedPoint.from_signed_int(value, total_bits, frac_bits))

    return list(map(to_float, values))
