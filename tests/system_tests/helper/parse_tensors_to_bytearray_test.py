import torch

from elasticai.creator.arithmetic import FxpArithmetic, FxpParams
from tests.system_tests.helper.parse_tensors_to_bytearray import (
    parse_bytearray_to_fxp_tensor,
    parse_fxp_tensor_to_bytearray,
)


def test_tensors_to_bytearray():
    torch.manual_seed(0)
    total_bits = 8
    frac_bits = 2
    fxp_conf = FxpArithmetic(
        FxpParams(total_bits=total_bits, frac_bits=frac_bits, signed=True)
    )
    tensor = fxp_conf.as_rational(fxp_conf.cut_as_integer(torch.randn(2, 3, 4)))
    result = parse_fxp_tensor_to_bytearray(tensor, total_bits, frac_bits)
    expected = [
        bytearray(b"\xfc\x03\x01\xfc\x02\x00\xff\xff\x05\xff\xf8\x01"),
        bytearray(b"\x00\x03\x02\x03\x05\xff\x00\x08\xfd\x00\x00\xf7"),
    ]
    assert result == expected


def test_bytearray_to_tensor():
    input = [
        bytearray(b"\xfc\x03\x01\xfc\x02\x00\xff\xff\x05\xff\xf8\x01"),
        bytearray(b"\x00\x03\x02\x03\x05\xff\x00\x08\xfd\x00\x00\xf7"),
    ]
    total_bits = 8
    frac_bits = 2
    torch.manual_seed(0)
    fxp_conf = FxpArithmetic(
        FxpParams(total_bits=total_bits, frac_bits=frac_bits, signed=True)
    )
    dimensions = (2, 3, 4)
    expected = fxp_conf.as_rational(fxp_conf.cut_as_integer(torch.randn(dimensions)))
    result = parse_bytearray_to_fxp_tensor(input, total_bits, frac_bits, dimensions)
    assert torch.equal(result, expected)
