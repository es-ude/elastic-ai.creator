import torch.nn
from elasticai.creator.blocks import QConv2d_block
from elasticai.creator.layers import ChannelShuffle

"Creation of multiple blocks of layers. While it could be done in a single module. Each individual block needs to be precalculated separatelly, so it is easier to represent it this way."

def generate_conv2d_sequence_with_width(in_channels, out_channels, channel_width,kernel_size, weight_quantization, activation):
    """
    This will generate a sequential model composed of multiple layers, each a weight with channel width length. 
    After the first block the channels are shuffled so the information of each output channel is composed from one in each input channel.
    eg: 256x256 1x1 convolution with channel width of 8. Each channel of the first layer will join information of 8 points  making 256/8 = 32 parts per out channel.
    So 32 groups and 256(out) = 256*32 output channels in the first layer. the second layer will again join 8  of those 32 resulting in 256*4
    groups and out channels. The last to keep the width will have a smaller number of groups e.g 256*4/8 = 128 groups. This has the effect of making very small kernels, 
    being able to be represented by a much smaller amount of LUTs than a 256:1 mapping.  
    
    Args:
        in_channels: 
        out_channels: 
        channel_width: 
        kernel_size: 
        weight_quantization: 
        activation: 

    Returns:

    """
    layers = []
    if in_channels < channel_width:
        raise ValueError("Channel width cannot be bigger than the number of input channels")
    next_in_channels = in_channels
    next_groups = (in_channels) // channel_width
    next_out_channels = out_channels * next_groups
    
    while (next_out_channels>= out_channels):
        layers.append(QConv2d_block(in_channels=next_in_channels, out_channels=next_out_channels, kernel_size=kernel_size,
                                        activation=activation, conv_quantizer=weight_quantization, groups=next_groups))
        if len(layers) == 1:
            layers.append(ChannelShuffle(groups=next_groups))
        if next_out_channels == out_channels:
            break
        next_in_channels = next_out_channels
        next_groups = next_groups // channel_width if len(layers)>2 else next_out_channels//channel_width
        next_out_channels = max(next_groups,out_channels)
            
    return torch.nn.Sequential(*layers)
