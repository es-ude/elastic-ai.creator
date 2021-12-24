class FixedPointConverter:
    """
    Create a fixed point representation as an int data type.

    We might want to have this create its own type `FixedPointNumber` in
    the future. That way we could make sure that the conversion is idempotent
    for numbers that are fixed point already.
    """

    def __init__(self, bits_used_for_fraction: int):
        self.bits_used_for_fraction = bits_used_for_fraction

    @property
    def one(self) -> int:
        return 1 << self.bits_used_for_fraction

    def from_float(self, x: float) -> int:
        x_tmp = float(x)
        x_tmp = x_tmp * self.one
        if not x_tmp.is_integer():
            raise ValueError(
                f"{x} not convertible to fixed point number using {self.bits_used_for_fraction} bits for fractional part"
            )
        return int(x_tmp)
