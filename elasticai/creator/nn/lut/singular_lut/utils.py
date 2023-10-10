from typing import Iterable, Sequence

import torch
from torch import tensor


def vhd_when(pair):
    a = pair[0]
    b = pair[1]
    return f"when {a} => y <= {b};"


def vhd_lut_cases(io_pairs):
    def to_vhd_text(io_pairs):
        """
        Args:
            io_pairs: is a sequence of (input, output) pairs.
        input and output have the form (B, C, K, N)
        [B: batch size, C: input channels,
         K: kernel size, N: number of sample points]"""

        def flatten(input):
            for channel in input:
                for value in channel:
                    yield int(value)

        for input, output in io_pairs:
            yield "".join(map(str, flatten(input))), "".join(map(str, flatten(output)))

    for io in to_vhd_text(io_pairs):
        yield vhd_when(io)
    # noinspection PyRedundantParentheses
    yield "when others => y <= '0';"


def generate_input_tensor(in_channels, kernel_size):
    length = in_channels * kernel_size
    elements = tuple(tensor([-1.0, 1.0]) for _ in range(length))
    return torch.tensor(torch.cartesian_prod(*elements)).reshape(
        2**length, in_channels, kernel_size
    )


def _to_bit_vector(x):
    return (x + 1) / 2


def generate_io_pair(module):
    x = generate_input_tensor(module.in_channels, module.kernel_size)
    y = module(x)
    x = _to_bit_vector(x)
    y = _to_bit_vector(y)
    y = y.tolist()
    x = x.tolist()
    return x, y
