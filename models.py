import utils
from model import create_model
import torch.nn as nn

def get_default(input_depth=3, output_depth=3, pad='zero'):
    return create_model(downsample_channels = [128, 128, 128, 128, 128],
                        upsample_channels = [128, 128, 128, 128, 128],
                        skip_channels = [4, 4, 4, 4, 4],
                        input_channel_size = input_depth,
                        output_channel_size = output_depth,
                        upsample_mode='bilinear',
                        activation_function=nn.LeakyReLU(0.2, inplace=True),
                        padding_type=pad
                        )

def get_simple(input_depth=3, output_depth=3, pad='zero'):
    return create_model(downsample_channels = [32, 32],
                        upsample_channels = [32, 32],
                        skip_channels = [0, 0],
                        input_channel_size = input_depth,
                        output_channel_size = output_depth,
                        upsample_mode='bilinear',
                        activation_function=nn.LeakyReLU(0.2, inplace=True),
                        padding_type=pad
                        )
def get_no_skip(input_depth=3, output_depth=3, pad='zero'):
    return create_model(downsample_channels = [128, 128, 128, 128, 128],
                        upsample_channels = [128, 128, 128, 128, 128],
                        skip_channels = [0, 0, 0, 0, 0],
                        input_channel_size = input_depth,
                        output_channel_size = output_depth,
                        upsample_mode='bilinear',
                        activation_function=nn.LeakyReLU(0.2, inplace=True),
                        padding_type=pad
                        )


def get_large_skip(input_depth=3, output_depth=3, pad='zero'):
    return create_model(downsample_channels = [128, 128, 128, 128, 128],
                        upsample_channels = [128, 128, 128, 128, 128],
                        skip_channels = [64, 64, 64, 64, 64],
                        input_channel_size = input_depth,
                        output_channel_size = output_depth,
                        upsample_mode='bilinear',
                        activation_function=nn.LeakyReLU(0.2, inplace=True),
                        padding_type=pad
                        )

def get_inc_no_skip(input_depth=3, output_depth=3, pad='zero'):
    return create_model(downsample_channels = [16, 32, 64, 128, 128],
                        upsample_channels = [16, 32, 64, 128, 128],
                        skip_channels = [0, 0, 0, 0, 0],
                        input_channel_size = input_depth,
                        output_channel_size = output_depth,
                        upsample_mode='bilinear',
                        activation_function=nn.LeakyReLU(0.2, inplace=True),
                        padding_type=pad
                        )


def get_inc_dec_filter_size(input_depth=3, output_depth=3, pad='zero'):
    return create_model(downsample_channels = [16, 32, 64, 128, 128],
                        upsample_channels = [16, 32, 64, 128, 128],
                        skip_channels = [0, 0, 0, 0, 0],
                        filter_size_down = [7, 5, 5, 3, 3],
                        filter_size_up = [7, 5, 5, 3, 3],
                        input_channel_size = input_depth,
                        output_channel_size = output_depth,
                        upsample_mode='bilinear',
                        activation_function=nn.LeakyReLU(0.2, inplace=True),
                        padding_type=pad
                        )
