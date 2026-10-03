from torch import Tensor

from .delta import Delta


class ConsecutiveDelta(Delta):
    def delta(self, input: Tensor, in_place: bool = True) -> Tensor:
        original_shape = input.shape
        workingcopy: Tensor = input if in_place else input.clone()
        workingcopy = workingcopy.flatten()
        workingcopy[1:] -= workingcopy[:-1].clone()
        return workingcopy.reshape(original_shape)

    def reconstruct(self, input: Tensor, in_place: bool = True) -> Tensor:
        original_shape = input.shape
        workingcopy: Tensor = input if in_place else input.clone()
        workingcopy = workingcopy.flatten()
        workingcopy.cumsum_(0)
        return workingcopy.reshape(original_shape)
