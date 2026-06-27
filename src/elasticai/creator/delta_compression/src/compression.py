from torch import Tensor

from .configuration import DeltaConf


def __compress_without_saturation(conf: DeltaConf, input: Tensor) -> Tensor:
    def bitmask(delta_bits: int, delta_offset: int) -> int:
        bitmask = 0
        for bit_index in range(delta_bits - 1):
            bitmask |= 1 << (bit_index + delta_offset)
        return bitmask

    negatvive_indexes = input < 0
    input.abs_()
    input &= bitmask(delta_bits=conf.width, delta_offset=conf.offset)
    input[negatvive_indexes] *= -1

    return input


def __compress_with_saturation(conf: DeltaConf, input: Tensor) -> Tensor:
    negainputive_indexes = input < 0
    input.abs_()
    input.clamp_(
        min=2 ** (conf.offset) if conf.offset > 0 else 0,
        max=2 ** (conf.width - 1 + conf.offset) - 1,
    )
    input[negainputive_indexes] *= -1

    return input


def clamp(conf: DeltaConf, input: Tensor, in_place: bool = True) -> Tensor:
    original_shape = input.shape
    workingcopy: Tensor = input if in_place else input.clone()
    workingcopy = workingcopy.flatten()

    if conf.saturate:
        workingcopy = __compress_with_saturation(conf, workingcopy)
    else:
        workingcopy = __compress_without_saturation(conf, workingcopy)

    return workingcopy.reshape(original_shape)
