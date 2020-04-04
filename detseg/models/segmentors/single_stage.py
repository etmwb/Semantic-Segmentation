import time
import math
import torch
from torch.nn.functional import interpolate

from detseg.core import seg_eval

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
        if hasattr(self, 'backbone_depth'):
            self.backbone_depth.init_weights(pretrained=pretrained)

    def extract_feat_hybrid(self, img):
        raise NotImplementedError

    def extract_feat(self, img):
        if self.with_depth:
            return self.extract_feat_hybrid(img)
        return self.backbone(img)[-1]

    def forward_dummy(self,
                      img,
                      gt_label=None,
                      depth=None,
                      HHA=None):
        return self.simple_test(img, gt_label, depth, HHA, logit_only=True)

    def forward_train(self,
                      img,
                      gt_label,
                      depth=None,
                      HHA=None,
                      cls_weight=None,
                      ignore_index=-100):
        img_h, img_w = img.size()[2:]

        img = self.pack(img, depth, HHA)

        feat_conv5 = self.extract_feat(img)
        logits = self.head(feat_conv5)
        logits_rescale = []
        for logit in logits:
            logits_rescale.append(interpolate(logit, size=(img_h, img_w), mode='bilinear', align_corners=True))

        losses = dict()

        loss_head = self.head.loss(logits_rescale, gt_label, cls_weight=cls_weight, ignore_index=ignore_index)
        losses.update(loss_head)

        return losses

    def simple_test(self,
                    img,
                    gt_label=None,
                    depth=None,
                    HHA=None,
                    ignore_index=-100,
                    logit_only=False):
        img_h, img_w = img.size()[2:]

        img = self.pack(img, depth, HHA)
        feat_conv5 = self.extract_feat(img)
        logit = self.head(feat_conv5)[0]
        logit = interpolate(logit, size=(img_h, img_w), mode='bilinear', align_corners=True)
        
        if logit_only:
            return logit
        else:
            return seg_eval(logit, gt_label, len(self.CLASSES), ignore_index)

    def aug_test(self,
                 img,
                 gt_label,
                 depth=None,
                 HHA=None,
                 scales=None,
                 ignore_index=-100
                 ):
        img_h, img_w = img.size()[2:]
        crop_size, long_side = min(img_h, img_w), max(img_h, img_w)
        stride_rate = 2.0 / 3.0
        stride = int(crop_size * stride_rate)

        with torch.cuda.device_of(img):
            logits = img.new().resize_(1, len(self.CLASSES), img_h, img_w).zero_().cuda()

        for scale in scales:
            # resize image to target scale
            long_size = int(math.ceil(long_side * scale))
            if img_h > img_w:
                height = long_size
                width = int(1.0 * img_w * long_size / img_h + 0.5)
                short_size = width
            else:
                width = long_size
                height = int(1.0 * img_h * long_size / img_w + 0.5)
                short_size = height
            resized_img, resized_depth, resized_HHA = [interpolate(tensor, size=(height, width), mode='bilinear', align_corners=True)
                                                       if tensor is not None else None for tensor in [img, depth, HHA]]

            if long_size <= crop_size:
                pad_img, pad_depth, pad_HHA = [pad_image(tensor, crop_size) if tensor is not None else None
                                               for tensor in [resized_img, resized_depth, resized_HHA]]
                outputs = self.simple_test(pad_img, None, pad_depth, pad_HHA, logit_only=True)
                flip_img, flip_depth, flip_HHA = [tensor.flip(-1) if tensor is not None else None
                                                  for tensor in [pad_img, pad_depth, pad_HHA]]
                outputs += self.simple_test(flip_img, None, flip_depth, flip_HHA, logit_only=True).flip(-1)
                outputs = outputs[:, :, :height, :width].exp()
            else:
                if short_size < crop_size:
                    # pad if needed
                    pad_img, pad_depth, pad_HHA = [pad_image(tensor, crop_size) if tensor is not None else None
                                                   for tensor in [resized_img, resized_depth, resized_HHA]]
                else:
                    pad_img, pad_depth, pad_HHA = resized_img, resized_depth, resized_HHA

                pad_h, pad_w = pad_img.size()[2:]
                # grid forward and normalize
                h_grids = int(math.ceil(1.0 * (pad_h - crop_size) / stride)) + 1
                w_grids = int(math.ceil(1.0 * (pad_w - crop_size) / stride)) + 1

                with torch.cuda.device_of(img):
                    outputs = img.new().resize_(1, len(self.CLASSES), pad_h, pad_w).zero_().cuda()
                    count_norm = img.new().resize_(1, 1, pad_h, pad_w).zero_().cuda()

                # grid evaluation
                for idh in range(h_grids):
                    for idw in range(w_grids):
                        h0 = idh * stride
                        w0 = idw * stride
                        h1 = min(h0 + crop_size, pad_h)
                        w1 = min(w0 + crop_size, pad_w)
                        crop_img, crop_depth, crop_HHA = [tensor[:, :, h0:h1, w0:w1] if tensor is not None else None
                                    for tensor in [pad_img, pad_depth, pad_HHA]]
                        # pad if needed
                        pad_crop_img, pad_crop_depth, pad_crop_HHA = [pad_image(tensor, crop_size) if tensor is not None else None
                                                                      for tensor in [crop_img, crop_depth, crop_HHA]]
                        output = self.simple_test(pad_crop_img, None, pad_crop_depth, pad_crop_HHA, logit_only=True)
                        flip_img, flip_depth, flip_HHA = [tensor.flip(-1) if tensor is not None else None
                                                          for tensor in [pad_crop_img, pad_crop_depth, pad_crop_HHA]]
                        output += self.simple_test(flip_img, None, flip_depth, flip_HHA, logit_only=True).flip(-1)
                        outputs[:, :, h0: h1, w0: w1] += output[:, :, 0: h1-h0, 0: w1-w0].exp()
                        count_norm[:, :, h0: h1, w0: w1] += 1
                assert((count_norm==0).sum()==0)
                outputs = outputs / count_norm
                outputs = outputs[:, :, :height, :width]

            logit = interpolate(outputs, size=(img_h, img_w), mode='bilinear', align_corners=True)
            logits += logit

        return seg_eval(logits, gt_label, len(self.CLASSES), ignore_index)


def pad_image(img, crop_size):
    b, _, h, w = img.size()
    padh = crop_size - h if h < crop_size else 0
    padw = crop_size - w if w < crop_size else 0
    img_pad = torch.nn.functional.pad(img, (0, padw, 0, padh), value=0)
    return img_pad
