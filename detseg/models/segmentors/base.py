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
        return hasattr(self, 'backbone_depth') and self.backbone_depth is not None

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

    def init_weights(self, pretrained=None):
        if pretrained is not None:
            logger = logging.getLogger()
            logger.info('load model from: {}'.format(pretrained))

    def forward_test(self, imgs, **kwargs):
        imgs = [imgs]
        for var, name in [(imgs, 'imgs')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))

        num_augs = len(imgs)
        # TODO: remove the restriction of imgs_per_gpu == 1 when prepared
        imgs_per_gpu = imgs[0].size(0)
        assert imgs_per_gpu == 1

        if num_augs == 1:
            return self.simple_test(imgs[0], **kwargs)
        else:
            return self.aug_test(imgs, **kwargs)

    @auto_fp16(apply_to=('img', ))
    def forward(self, img, return_loss=True, **kwargs):
        if return_loss:
            return self.forward_train(img, **kwargs)
        else:
            return self.forward_test(img, **kwargs)

    def show_result(self, data):
        # TODO: show segmentation result
        raise NotImplementedError