import torch
from torch.nn import Parameter


def random_mask_4d(
    out_channels: int,
    kernel_size: int | tuple,
    in_channels: int,
    groups: int,
    params_per_channel: int,
):
    """
    Creates a 4d mask with a  number of nonzero elements per out channels (index 0) equals to the params_per_channel randomly selected
    Args:
        out_channels:
        kernel_size:
        in_channels:
        groups:
        params_per_channel:

    Returns:

    """
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    mask = Parameter(
        torch.zeros((out_channels, in_channels // groups, *kernel_size)),
        requires_grad=False,
    )
    for i in range(out_channels):
        original_shape = mask.shape[1:]
        flattened_channel = mask[i].view(-1)
        random_indices = torch.randperm(flattened_channel.shape[0])
        flattened_channel[random_indices[:params_per_channel]] = 1
        mask[i] = torch.reshape(flattened_channel, original_shape)
    return mask


def fixed_offset_mask_4d(
    out_channels: int,
    kernel_size: int | tuple,
    in_channels: int,
    groups: int,
    axis_width: int,
    offset_axis=1,
):
    """
    Creates a 4d mask with an offset per out channel, on each channel each element of the offset indices part of the
    offset axis is set to 1. Can select more than 1 by setting an axis width.
    The offset will wrap over the axis so index % len(axis)

    Args:
        out_channels:
        kernel_size:
        in_channels:
        groups:
        axis_width:
        offset_axis:

    Returns:

    """
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    mask = Parameter(
        torch.zeros((out_channels, in_channels // groups, *kernel_size)),
        requires_grad=False,
    )

    for i in range(out_channels):

        axis_group_index = i % (mask.size()[offset_axis] // axis_width)
        axis_indices = list(
            map(lambda x: x + axis_group_index * axis_width, list(range(axis_width)))
        )
        if offset_axis == 1:
            mask[i, axis_indices, :, :] = 1
        if offset_axis == 2:
            mask[i, :, axis_indices, :] = 1
        if offset_axis == 3:
            mask[i, :, :, axis_indices] = 1

    return mask


class FixedOffsetMask4d(torch.nn.Module):
    def __init__(
        self,
        out_channels: int,
        kernel_size: int | tuple,
        in_channels: int,
        groups: int,
        axis_width: int,
        offset_axis=1,
    ):
        super().__init__()
        self.mask = fixed_offset_mask_4d(
            out_channels=out_channels,
            kernel_size=kernel_size,
            in_channels=in_channels,
            groups=groups,
            axis_width=axis_width,
            offset_axis=offset_axis,
        )

    def forward(self, input_values):
        return input_values * self.mask
