from typing import Iterable

import torch
from torch import Tensor


def torch1d_to_lutron(x: Tensor) -> Tensor:
    if x.dim() == 3:
        batch_size = x.size()[0]
        return x.permute(0, 2, 1).reshape(batch_size, -1)
    return x.permute(1, 0).reshape(-1)


def lutron_to_torch1d(x: Tensor, channels: int) -> Tensor:
    if x.dim() == 2:
        batch_size = x.size()[0]
        return x.view(batch_size, -1, channels).permute(0, 2, 1)
    return x.view(-1, channels).permute(1, 0)


def lutron_to_string(x: Tensor) -> str:
    return "".join(map(str, x.tolist()))


def string_to_lutron(x: str) -> Tensor:
    return torch.tensor((int(digit) for digit in x))


def string_to_torch1d(x: str, channels: int) -> Tensor:
    return lutron_to_torch1d(string_to_lutron(x), channels)


def group_torch1d(x: Tensor, groups: int) -> Tensor:
    """Convert a tensor (B, C, N) to (B, G, C/G, N)"""
    channel_dim = 1
    spatial_dim = 2
    batch_dim = 0
    if x.dim() < 3:
        raise ValueError

    channels = x.size()[channel_dim]
    kernel_size = x.size()[spatial_dim]
    batch_size = x.size()[batch_dim]
    in_group_size = channels // groups
    grouped = x.reshape(batch_size, groups, in_group_size, kernel_size)
    return grouped


def torch1d_input_tensor_batch_to_lutron_batches(x: Tensor, groups: int) -> Tensor:
    """
    Convert torch input tensor format (B, C, N) to (G, B, C/GxN).
    """
    grouped = group_torch1d(x, groups)
    input_lists_per_group = grouped.permute(1, 0, 2, 3)
    B = x.size()[0]
    return input_lists_per_group.reshape(groups, B, -1)


def grouped_tensor_batch_to_strings(inputs_by_group: Tensor) -> Iterable[Iterable[str]]:
    """`inputs_by_group` has the shape (G, B, C/GxN)"""

    result: list[list[str]] = []
    for group in inputs_by_group:
        tmp: list[str] = []
        for input in group:
            flattened_input = input.view(-1).tolist()
            tmp.append("".join(str(x) for x in flattened_input))
        result.append(tmp)
    return result


def torch1d_input_tensor_to_grouped_strings(
    x: Tensor, groups: int
) -> Iterable[Iterable[str]]:
    if x.dim() == 2:
        x = x.view(x.size(0), 1, x.size(1))
    elif x.dim() == 1:
        x = x.view(x.size(0), 1, 1)
    grouped = group_torch1d(x, groups)
    inputs_by_group = grouped.permute(1, 0, 2, 3)
    return grouped_tensor_batch_to_strings(inputs_by_group)
