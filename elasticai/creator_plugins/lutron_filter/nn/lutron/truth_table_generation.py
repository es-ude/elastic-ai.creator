from random import randint
from typing import Iterable

import torch
from torch import Tensor, tensor

from elasticai.creator.vhdl.code_generation.code_abstractions import (
    to_vhdl_binary_string,
)


def encode_as_number_per_channel(outs: Tensor):
    num_rows = outs.size()[0]
    if num_rows > 64:
        raise ValueError(
            "we can only encode 64 bits (six in channels), if you need more, split your truth table and encode each part separately"
        )

    multiplier = torch.tensor([2**i for i in reversed(range(num_rows))]).reshape(1, -1)
    for idx, number_as_bit_vector in enumerate(outs.T):
        number = torch.matmul(multiplier, number_as_bit_vector).item()
        yield number


def encode_as_number_per_row(outs: Tensor):
    num_out_bits = outs.size()[1]
    if num_out_bits > 64:
        raise ValueError("we can only encode 64 out channels")
    multiplier = torch.tensor([2**i for i in reversed(range(num_out_bits))])
    for number_as_bit_vector in outs:
        yield torch.matmul(multiplier, number_as_bit_vector).item()


def decode_from_number_per_channel_to_bit_vector_per_input(out_channels: Iterable[int]):
    out_channels = tuple(out_channels)
    num_truth_table_rows = 2 ** len(out_channels)

    def get_ith_bit(number, i):
        return (number // (2**i)) % 2

    for i in reversed(range(num_truth_table_rows)):
        bits = tuple(get_ith_bit(n, i) for n in out_channels)
        yield bits


def decode_from_number_per_channel_to_bitstrings(outs, in_channels):
    num_bits = len(outs)

    def to_string(bits):
        bit_string = "".join(map(str, bits))
        padding_length = num_bits - len(bit_string)
        return "".join(["0" * padding_length, bit_string])

    yield from map(
        to_string,
        decode_from_number_per_channel_to_bit_vector_per_input(outs),
    )


def group_tensors(ins, outs, groups):
    in_channels = ins.size()[1]
    out_channels = outs.size()[1]
    kernel_size = ins.size()[2]
    total = ins.size()[0]
    in_group_size = in_channels // groups
    out_group_size = out_channels // groups
    grouped_ins = ins.view(total, groups, in_group_size, kernel_size)
    grouped_outs = outs.view(total, groups, out_group_size, 1)
    return grouped_ins, grouped_outs


def convert_from_numbers_to_binary_logic(tensor):
    return ((2 * tensor + 1) / 2).to(torch.int)


def convert_grouped_tensors_to_strings(ins, outs):
    groups = ins.size()[1]
    total = ins.size()[0]
    for g in range(groups):
        _tmp = []
        for i in range(total):
            _tmp.append(
                (
                    "".join(map(str, ins[i, g].view(-1).tolist())),
                    "".join(map(str, outs[i, g].view(-1).tolist())),
                )
            )
        yield _tmp


def convert_to_list(ins, outs, groups):
    return list(convert_grouped_tensors_to_strings(*group_tensors(ins, outs, groups)))


def generate_input_tensor(in_channels, kernel_size, groups=1):
    length = in_channels // groups * kernel_size
    elements = tuple(tensor([-1.0, 1.0]) for _ in range(length))
    return (
        torch.cartesian_prod(*elements)
        .reshape(2**length, in_channels // groups, kernel_size)
        .repeat(1, groups, 1)
    )


def generate_io_pairs(in_bits: int, out_bits: int) -> dict[str, str]:
    pairs: dict[str, str] = {}
    for i in range(2**in_bits):
        s_i = to_vhdl_binary_string(i, number_of_bits=in_bits)
        output = randint(0, (2**out_bits) - 1)
        s_output = to_vhdl_binary_string(output, number_of_bits=out_bits)
        pairs[s_i] = s_output
    return pairs
