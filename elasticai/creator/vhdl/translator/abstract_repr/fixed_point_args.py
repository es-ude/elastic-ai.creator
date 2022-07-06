from dataclasses import dataclass


@dataclass
class FixedPointArgs:
    total_bits: int
    frac_bits: int
