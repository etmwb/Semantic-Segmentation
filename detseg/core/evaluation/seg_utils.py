import numpy as np
import torch


def seg_eval(logit, gt_label, num_classes, ignore_index):
    _, plabel = torch.max(logit, dim=1)
    plabel = plabel.cpu().numpy() + 1
    label = gt_label.cpu().numpy() + 1
    valid_pixels = np.sum(label != (ignore_index + 1))
    plabel = plabel * (label != (ignore_index + 1)).astype(plabel.dtype)
    intersection = plabel * (plabel == label)
    intersection_pixels = np.sum(intersection != 0)
    cls_plabel, _ = np.histogram(plabel, bins=num_classes, range=(1, num_classes))
    cls_label, _ = np.histogram(label, bins=num_classes, range=(1, num_classes))
    cls_intersection, _ = np.histogram(intersection, bins=num_classes, range=(1, num_classes))

    return dict(valid_pixels=valid_pixels, intersection_pixels=intersection_pixels,
                cls_label=cls_label, cls_plabel=cls_plabel, cls_intersection=cls_intersection)


def post_eval(results):
    pixAcc = results['intersection_pixels'] / results['valid_pixels']
    mAcc = np.mean(results['cls_intersection'] / results['cls_label'])
    print(results['cls_intersection'] / \
          (results['cls_plabel'] + results['cls_label'] - results['cls_intersection']))
    mIoU = np.mean(results['cls_intersection'] / \
                   (results['cls_plabel'] + results['cls_label'] - results['cls_intersection']))
    print('post_evaluate finished: pixAcc: {0:.3f} mAcc: {1:.3f} mIoU: {2:.3f}'.format(pixAcc, mAcc, mIoU))
    return results