from math import ceil, floor

from .precomputed_scalar_function import (
    _PrecomputedMonotonouslyIncreasingScalarFunction,
)


class HardSigmoid(_PrecomputedMonotonouslyIncreasingScalarFunction):
    def __init__(self, width: int, lower_bound_for_zero: int, upper_bound_for_one: int):
        self.lower_bound_for_zero = lower_bound_for_zero
        self.upper_bound_for_one = upper_bound_for_one
        self.slope = 1 / (self.upper_bound_for_one - self.lower_bound_for_zero)
        self.zero_point = -(self.slope * self.lower_bound_for_zero)
        super().__init__(
            name="hard_sigmoid",
            width=width,
            function=self._function,
            inputs=list(range(lower_bound_for_zero, upper_bound_for_one)),
        )

    @staticmethod
    def _rounding_half_to_even(x: float) -> int:
        distance_to_ceil = ceil(x) - x
        distance_to_floor = x - floor(x)
        if distance_to_floor < distance_to_ceil:
            return floor(x)
        if distance_to_ceil < distance_to_floor:
            return ceil(x)
        else:
            if ceil(x) % 2 == 0:
                return ceil(x)
            else:
                return floor(x)

    def _function(self, x: int) -> int:
        if x < self.lower_bound_for_zero:
            return 0
        if x > self.upper_bound_for_one:
            return 1
        return self._rounding_half_to_even(x * self.slope + self.zero_point)
