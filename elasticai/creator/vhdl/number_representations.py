class FixedPointBuilder:
    def __init__(self, bits_used_for_fraction: int):
        self.bits_used_for_fraction = bits_used_for_fraction

    @property
    def one(self) -> int:
        return 1 << self.bits_used_for_fraction

    def build(self, x: float) -> int:
        x_tmp = float(x)
        x_tmp = x_tmp * self.one
        if not x_tmp.is_integer():
            raise ValueError(
                f"{x} not convertible to fixed point number using {self.bits_used_for_fraction} bits for fractional part"
            )
        return int(x_tmp)
