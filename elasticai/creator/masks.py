import math
from typing import Union, Tuple

import torch
from torch.nn import Parameter


def randomMask4D(out_channels:int, kernel_size: Union[int,Tuple], in_channels:int, groups:int, params_per_channel:int):


    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    mask = Parameter(torch.zeros((out_channels, in_channels // groups, *kernel_size)),
                          requires_grad=False)
    for i in range(out_channels):
        original_shape = mask.shape[1:]
        flattened_channel = mask[i].view(-1)
        random_indices = torch.randperm(flattened_channel.shape[0])
        flattened_channel[random_indices[:params_per_channel]] = 1
        mask[i] =torch.reshape(flattened_channel,original_shape)
    return mask



def fixedOffsetMask4D(out_channels:int, kernel_size: Union[int, Tuple], in_channels:int, groups:int, channel_width:int, offset_axis = 1):
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    mask = Parameter(torch.zeros((out_channels, in_channels // groups, *kernel_size)),
                     requires_grad=False)
    

    for i in range(out_channels):
        if offset_axis ==1:
            channel_group_index = i % (in_channels //(groups*channel_width))  
            channel_indices = list(map(lambda x : x + channel_group_index * channel_width,list(range(channel_width))))
            mask[i,channel_indices,:,:] = 1
        else:
            if (channel_width != in_channels//groups) & (channel_width != 1):
                raise NotImplementedError("Please set channel_width == in_channels//groups or 1, offsets over 2 axis are not supported, leverage the groups parameter instead")
            axis_index = i % mask.size()[offset_axis]
            if offset_axis == 2:
                mask[i,:, axis_index,:] = 1
            if offset_axis == 3:
                mask[i, :, :, axis_index] = 1
    
    return mask

