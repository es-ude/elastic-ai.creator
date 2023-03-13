from typing import Callable

from elasticai.creator.vhdl.number_representations import FixedPoint


def derive_fixed_point_params_from_factory(
    fixed_point_factory: Callable[[float], FixedPoint]
) -> tuple[int, int]:
    dummy_value = fixed_point_factory(0)
    return dummy_value.total_bits, dummy_value.frac_bits
