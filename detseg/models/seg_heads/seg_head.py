import os.path as osp
import numpy as np

import torch
import torch.nn as nn

from detseg.core import (force_fp32, multi_apply)
from ..builder import build_loss
from ..registry import SEG_HEADS

@SEG_HEADS.register_module
class SegHead(nn.Module):

    def __init__(self,
                 loss_cfg=None):
        super(SegHead, self).__init__()
        self.loss_head = build_loss(loss_cfg)

    @force_fp32(apply_to=('cls_logits', ))
    def loss(self,
             cls_logits,
             gt_labels,
             cls_weight=None,
             ignore_index=-100):

        if cls_weight is not None:
            if isinstance(cls_weight, str):
                assert osp.isfile(cls_weight)
                with open(cls_weight, 'r') as f:
                    cls_weight = list(map(lambda x: float(x.strip()), f.readlines()))
            elif isinstance(cls_weight, list):
                pass
            else:
                raise NotImplementedError('Unsupported cls weight type {}'.format(type(cls_weight).__name__))
            cls_weight = torch.from_numpy(np.array(cls_weight)).float()

        loss_head = [None for _ in range(len(cls_logits))]
        for i, logit in enumerate(cls_logits):
            loss_head[i] = self.loss_head(logit, gt_labels, cls_weight=cls_weight, ignore_index=ignore_index)

        return dict(
            loss_head=sum(loss_head))