import math

import torch.nn as nn
from mmcv.cnn import constant_init, kaiming_init
from torch.nn.modules.batchnorm import _BatchNorm

from ..registry import BACKBONES
from ..utils import ConvModule


@BACKBONES.register_module
class PointNet(nn.Module):
    def __init__(self,
                 in_channels=3,
                 num_stages=5,
                 out_channels=(64, 64, 64, 128, 256),
                 strides=(2, 2, 2, 1, 1),
                 out_indices=(1, 2, 3, 4),
                 norm_cfg=dict(type='BN', requires_grad=True)):
        super(PointNet, self).__init__()
        assert num_stages == len(out_channels) == len(strides)
        self.out_indices = out_indices

        self.layers = []
        out_channels = (in_channels, ) + out_channels
        for i in range(num_stages):
            layer = ConvModule(out_channels[i],
                               out_channels[i+1],
                               kernel_size=1,
                               stride=strides[i],
                               norm_cfg=norm_cfg)
            layer_name = 'layer{}'.format(i+1)
            self.add_module(layer_name, layer)
            self.layers.append(layer_name)

    def init_weights(self, pretrained=None):
        # ignore args "pretrained" here
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        outs = []
        for i, layer_name in enumerate(self.layers):
            layer = getattr(self, layer_name)
            x = layer(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)