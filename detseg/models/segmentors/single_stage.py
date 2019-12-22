import numpy as np
import torch
from torch.nn.functional import interpolate

from ..registry import SEGMENTORS
from .base import BaseSegmentor
from .. import builder

SEGMENTORS.register_module
class SingleStageSegmentor(BaseSegmentor):

    def __init__(self,
                 backbone,
                 head,
                 backbone_depth=None,
                 pretrained=None):
        super(SingleStageSegmentor, self).__init__()
        self.backbone = builder.build_backbone(backbone)
        self.head = builder.build_seg_head(head)

        if backbone_depth is not None:
            self.backbone_depth = builder.build_backbone(backbone_depth)

        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        super(SingleStageSegmentor, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        self.head.init_weights()
        if self.with_depth:
            self.backbone_depth.init_weights(pretrained=pretrained)

    def extract_feat_hybrid(self, img):
        raise NotImplementedError

    def extract_feat(self, img):
        if self.with_depth:
            return self.extract_feat_hybrid(img)
        return self.backbone(img)[-1]

    def forward_train(self,
                      img,
                      gt_label,
                      depth=None,
                      HHA=None,
                      cls_weight=None,
                      ignore_index=-100):
        assert not (depth is not None and HHA is not None)
        img_h, img_w = img.size()[2:]
        if depth is not None:
            img = [img, depth]
        if HHA is not None:
            img = [img, HHA]

        x = self.extract_feat(img)
        x = self.head(x)

        cls_logits_new = []
        for cls_logit in x:
            cls_logits_new.append(interpolate(cls_logit, size=(img_h, img_w), mode='bilinear', align_corners=True))

        losses = dict()

        loss_head = self.head.loss(cls_logits_new, gt_label, cls_weight=cls_weight, ignore_index=ignore_index)
        losses.update(loss_head)

        return losses

    def simple_test(self,
                    img,
                    gt_label,
                    depth=None,
                    HHA=None,
                    cls_weight=None,
                    ignore_index=-100
                    ):
        assert not (depth is not None and HHA is not None)
        img_h, img_w = img.size()[2:]
        if depth is not None:
            img = [img, depth]
        if HHA is not None:
            img = [img, HHA]

        x = self.extract_feat(img)
        x = self.head(x)[0]
        x = interpolate(x, size=(img_h, img_w), mode='bilinear', align_corners=True)

        num_classes = len(self.CLASSES)

        _, plabel = torch.max(x, dim=1)
        plabel = plabel.cpu().numpy() + 1
        label = gt_label.cpu().numpy() + 1
        valid_pixels = np.sum(label != (ignore_index+1))
        plabel = plabel * (label != (ignore_index+1)).astype(plabel.dtype)
        intersection = plabel * (plabel == label)
        intersection_pixels = np.sum(intersection != 0)
        cls_plabel, _ = np.histogram(plabel, bins=num_classes, range=(1, num_classes))
        cls_label, _ = np.histogram(label, bins=num_classes, range=(1, num_classes))
        cls_intersection, _ = np.histogram(intersection, bins=num_classes, range=(1, num_classes))

        return dict(valid_pixels=valid_pixels, intersection_pixels=intersection_pixels,
                    cls_label=cls_label, cls_plabel=cls_plabel, cls_intersection=cls_intersection)

    def aug_test(self, imgs, **kwargs):
        pass