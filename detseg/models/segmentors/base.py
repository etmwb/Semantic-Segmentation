import logging
from abc import ABCMeta, abstractmethod

import torch.nn as nn

from detseg.core import auto_fp16


class BaseSegmentor(nn.Module):
    """Base class for segmentors"""

    __metaclass__ = ABCMeta

    def __init__(self):
        super(BaseSegmentor, self).__init__()
        self.fp16_enabled = False

    @property
    def with_depth(self):
        return self.backbone.dcn is not None and 'depth' in self.backbone.dcn['dcn_type']

    @abstractmethod
    def extract_feat(self, imgs):
        pass

    @abstractmethod
    def forward_train(self, imgs, **kwargs):
        pass

    @abstractmethod
    def simple_test(self, img, **kwargs):
        pass

    @abstractmethod
    def aug_test(self, imgs, **kwargs):
        pass

    def pack(self, img, depth=None, HHA=None, PC=None):
        assert not (depth is not None and HHA is not None)
        if depth is not None:
            img = [img, depth]
        if HHA is not None:
            if PC is not None: 
                img = [img, (HHA, PC)]
            else:
                img = [img, HHA]
        return img

    def init_weights(self, pretrained=None):
        if pretrained is not None:
            logger = logging.getLogger()
            logger.info('load model from: {}'.format(pretrained))

    def forward_test(self, img, **kwargs):

        # TODO: remove the restriction of imgs_per_gpu == 1 when prepared
        imgs_per_gpu = img.size(0)
        assert imgs_per_gpu == 1

        if not kwargs.get('scales', None):
            if 'scales' in kwargs: kwargs.pop('scales')
            return self.simple_test(img, **kwargs)
        else:
            return self.aug_test(img, **kwargs)

    @auto_fp16(apply_to=('img', ))
    def forward(self, img, return_loss=True, **kwargs):
        if 'rescale' in kwargs:
            kwargs.pop('rescale')

        if return_loss:
            return self.forward_train(img, **kwargs)
        else:
            return self.forward_test(img, **kwargs)
