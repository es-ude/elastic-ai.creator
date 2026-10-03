from torch import Tensor

from .compression import Compression


class SaturatedCompression(Compression):
    def __init__(self, width: int, offset: int) -> None:
        self.width = width
        self.offset = offset

    def compress(self, input: Tensor, in_place: bool = True) -> Tensor:
        original_shape = input.shape
        workingcopy: Tensor = input if in_place else input.clone()
        workingcopy = workingcopy.flatten()

        negative_indexes = workingcopy < 0
        workingcopy.abs_()
        workingcopy.clamp_(
            min=0 if self.offset == 0 else 2 ** (self.offset),
            max=2 ** (self.width - 1 + self.offset) - 1,
        )
        workingcopy[negative_indexes] *= -1

        return workingcopy.reshape(original_shape)

    def inflate(self, input: Tensor, in_place: bool = True) -> Tensor:
        workingcopy: Tensor = input if in_place else input.clone()
        return workingcopy
