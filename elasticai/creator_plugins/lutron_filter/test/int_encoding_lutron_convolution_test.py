import pytest
import torch

from elasticai.creator_plugins.lutron_filter.torch.lutron_modules.int_encoding_lutron_convolution import (
    convert_lutron_bits_to_ints_as_two_complements,
)


@pytest.mark.parametrize(
    "inputs, expected, in_channels",
    (
        ((1, -1), (-2,), 1),
        ((-1, 1, -1), (2,), 1),
        ((-1, 1, -1, -1, 1), (9,), 1),
        ((-1, 1, 1, 1, -1, -1), (3, -4), 2),
    ),
)
def test_convert_lutron_bits_to_ints(inputs, expected, in_channels):
    inputs = torch.tensor(inputs, dtype=torch.float32)
    inputs = inputs.reshape((1, len(inputs), 1))
    result = convert_lutron_bits_to_ints_as_two_complements(
        inputs,
        in_channels=in_channels,
    )
    result = result.view(-1)
    result = tuple(result.tolist())
    assert result == expected
