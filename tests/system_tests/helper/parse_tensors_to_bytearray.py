from torch import Tensor, flatten

from elasticai.creator.arithmetic import (
    FxpArithmetic,
    FxpParams,
)


def parse_fxp_tensor_to_bytearray(
    tensor: Tensor, total_bits: int, frac_bits: int
) -> list[bytearray]:
    tensor = flatten(tensor.permute([0, 2, 1]), start_dim=1)
    fxp_params = FxpParams(total_bits=total_bits, frac_bits=frac_bits, signed=True)
    fxp_config = FxpArithmetic(fxp_params)
    ints = fxp_config.cut_as_integer(tensor).tolist()
    data = list()
    for i, batch in enumerate(ints):
        data.append(bytearray())
        for item in batch:
            item_as_bytes = int(item).to_bytes(1, byteorder="big", signed=True)
            data[i].extend(item_as_bytes)
    return data


def parse_bytearray_to_fxp_tensor(
    data: list[bytearray], total_bits: int, frac_bits: int, dimensions: tuple
) -> Tensor:
    fxp_params = FxpParams(total_bits=total_bits, frac_bits=frac_bits, signed=True)
    fxp_config = FxpArithmetic(fxp_params)
    rationals = list()
    for i, batch in enumerate(data):
        rationals.append(list())
        my_batch = batch.hex(sep=" ", bytes_per_sep=1).split(" ")
        for item in my_batch:
            item_as_bytes = bytes.fromhex(item)
            item_as_int = int.from_bytes(item_as_bytes, byteorder="big", signed=True)
            item_as_rational = fxp_config.as_rational(item_as_int)
            rationals[i].append(item_as_rational)
    tensor = Tensor(rationals)
    tensor = tensor.unflatten(1, (dimensions[2], dimensions[1]))
    tensor = tensor.transpose(1, 2)
    return tensor
