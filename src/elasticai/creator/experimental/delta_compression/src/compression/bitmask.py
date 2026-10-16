from torch import Tensor

from .compression import Compression


class BitMaskCompression(Compression):
    def __init__(self, width: int, offset: int) -> None:
        self.width = width
        self.offset = offset

    def compress(self, input: Tensor, in_place: bool = True) -> Tensor:
        original_shape = input.shape
        workingcopy: Tensor = input if in_place else input.clone()
        workingcopy = workingcopy.flatten()

        def bitmask(delta_bits: int, delta_offset: int) -> int:
            bitmask = 0
            for bit_index in range(delta_bits - 1):
                bitmask |= 1 << (bit_index + delta_offset)
            return bitmask

        negatvive_indexes = workingcopy < 0
        workingcopy.abs_()
        workingcopy &= bitmask(delta_bits=self.width, delta_offset=self.offset)
        workingcopy[negatvive_indexes] *= -1

        return workingcopy.reshape(original_shape)

    def inflate(self, input: Tensor, in_place: bool = True) -> Tensor:
        workingcopy: Tensor = input if in_place else input.clone()
        return workingcopy
