from torch import Tensor

from .delta import Delta


class FixedReferenceDelta(Delta):
    def delta(self, input: Tensor, in_place: bool = True) -> Tensor:
        original_shape = input.shape
        workingcopy: Tensor = input if in_place else input.clone()
        workingcopy = workingcopy.flatten()
        workingcopy[1:] -= workingcopy[0]
        return workingcopy.reshape(original_shape)

    def reconstruct(self, input: Tensor, in_place: bool = True) -> Tensor:
        original_shape = input.shape
        workingcopy: Tensor = input if in_place else input.clone()
        workingcopy = workingcopy.flatten()
        workingcopy[1:] += workingcopy[0]
        return workingcopy.reshape(original_shape)
