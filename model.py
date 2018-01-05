import torch
import torch.nn as nn
from concat import Concat

def get_conv_layer(input_size, output_size, filter_size, stride=1, bias=True, padding_type='zero'):
    conv_layer = nn.Sequential()
    padding = int((filter_size - 1) / 2)
    if padding_type == 'reflection':
        conv_layer.add(nn.ReflectionPad2d(padding))
        padding = 0
    conv_layer.add(nn.Conv2d(input_size, output_size, filter_size, stride, padding=padding, bias=bias))
    return conv_layer

def create_model(downsample_channels,
                 upsample_channels,
                 skip_channels,
                 input_channel_size,
                 output_channel_size,
                 filter_size_down=3,
                 filter_size_up=3,
                 filter_size_skip=1,
                 activation_function=nn.LeakyReLU(0.2, inplace=True),
                 use_sigmoid=True,
                 upsample_mode = 'nearest',
                 padding_type = 'zero',
                 need1x1 = True
                 ):

    assert len(downsample_channels) == len(upsample_channels) == len(skip_channels)
    assert isinstance(downsample_channels, list)
    assert isinstance(upsample_channels, list)
    assert isinstance(skip_channels, list)

    layer_length = len(downsample_channels)
    last_layer = layer_length - 1

    if isinstance(filter_size_down, int):
        filter_size_down = [filter_size_down] * layer_length

    if isinstance(filter_size_up, int):
        filter_size_up = [filter_size_up] * layer_length

    if isinstance(filter_size_skip, int):
        filter_size_skip = [filter_size_skip] * layer_length

    model = nn.Sequential()
    layer = model
    input_size = input_channel_size

    for idx, d_channels in enumerate(downsample_channels):
        print(f'layer {idx}')
        next_layer = nn.Sequential()
        skip_layer = nn.Sequential()

        # Skip
        if skip_channels[idx] is not 0:
            layer.add(Concat(1, skip_layer, next_layer))
            skip_layer.add(get_conv_layer(input_size, skip_channels[idx], filter_size_skip[idx], padding_type=padding_type))
            skip_layer.add(nn.BatchNorm2d(skip_channels[idx]))
            skip_layer.add(activation_function)
        else:
            layer.add(next_layer)

        # Downsample
        next_layer.add(get_conv_layer(input_size, d_channels, filter_size_down[idx], stride=2, padding_type=padding_type))
        next_layer.add(nn.BatchNorm2d(d_channels))
        next_layer.add(activation_function)

        next_layer.add(get_conv_layer(d_channels, d_channels, filter_size_down[idx], padding_type=padding_type))
        next_layer.add(nn.BatchNorm2d(d_channels))
        next_layer.add(activation_function)

        next_next_layer = nn.Sequential()

        if idx == last_layer:
            # The deepest
            k = d_channels
        else:
            next_layer.add(next_next_layer)
            k = upsample_channels[idx + 1]

        # Upsample
        next_layer.add(nn.Upsample(scale_factor=2, mode=upsample_mode))

        bn_channels = upsample_channels[idx + 1] if (idx < last_layer) else d_channels
        layer.add(nn.BatchNorm2d(skip_channels[idx] + bn_channels))

        # Include skipping layer
        layer.add(get_conv_layer(skip_channels[idx] + k, upsample_channels[idx], filter_size_up[idx], padding_type=padding_type))
        layer.add(nn.BatchNorm2d(upsample_channels[idx]))
        layer.add(activation_function)

        # Upsample with filter size 1x1
        if need1x1:
            layer.add(get_conv_layer(upsample_channels[idx], upsample_channels[idx], filter_size=1, padding_type=padding_type))
            layer.add(nn.BatchNorm2d(upsample_channels[idx]))
            layer.add(activation_function)

        input_size = d_channels
        layer = next_next_layer

    model.add(get_conv_layer(upsample_channels[0], output_channel_size, filter_size=1, padding_type=padding_type))
    if use_sigmoid:
        model.add(nn.Sigmoid())

    return model





