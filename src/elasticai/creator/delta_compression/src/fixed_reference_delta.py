from torch import Tensor


def fixed_reference_delta(input: Tensor, in_place: bool = True) -> Tensor:
    original_shape = input.shape
    workingcopy: Tensor = input if in_place else input.clone()
    workingcopy = workingcopy.flatten()
    workingcopy[1:] -= workingcopy[0]
    return workingcopy.reshape(original_shape)


def reverse_fixed_reference_delta(input: Tensor, in_place: bool = True) -> Tensor:
    original_shape = input.shape
    workingcopy: Tensor = input if in_place else input.clone()
    workingcopy = workingcopy.flatten()
    workingcopy[1:] += workingcopy[0]
    return workingcopy.reshape(original_shape)
