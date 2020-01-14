import argparse
import os
import re
import cv2
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as mcol
import matplotlib.pyplot as plt
import shutil
import tempfile

import mmcv
import torch
import torch.distributed as dist
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, load_checkpoint, obj_from_dict

from detseg.apis import init_dist, get_root_logger
from detseg.core import coco_eval, results2json, wrap_fp16_model
from detseg.datasets import build_dataloader, build_dataset
from detseg.models import build_segmentor, build_detector
from detseg.core.utils.dist_utils import allreduce_grads


def build_optimizer(model, optimizer_cfg):
    """Build optimizer from configs.

    Args:
        model (:obj:`nn.Module`): The model with parameters to be optimized.
        optimizer_cfg (dict): The config dict of the optimizer.
            Positional fields are:
                - type: class name of the optimizer.
                - lr: base learning rate.
            Optional fields are:
                - any arguments of the corresponding optimizer type, e.g.,
                  weight_decay, momentum, etc.
                - paramwise_options: a dict with 3 accepted fileds
                  (bias_lr_mult, bias_decay_mult, norm_decay_mult).
                  `bias_lr_mult` and `bias_decay_mult` will be multiplied to
                  the lr and weight decay respectively for all bias parameters
                  (except for the normalization layers), and
                  `norm_decay_mult` will be multiplied to the weight decay
                  for all weight and bias parameters of normalization layers.

    Returns:
        torch.optim.Optimizer: The initialized optimizer.

    Example:
        >>> model = torch.nn.modules.Conv1d(1, 1, 1)
        >>> optimizer_cfg = dict(type='SGD', lr=0.01, momentum=0.9,
        >>>                      weight_decay=0.0001)
        >>> optimizer = build_optimizer(model, optimizer_cfg)
    """
    if hasattr(model, 'module'):
        model = model.module

    optimizer_cfg = optimizer_cfg.copy()
    paramgroup_options = optimizer_cfg.pop('paramgroup_options')
    if paramgroup_options is not None:
        param_list, base_lr = [], optimizer_cfg['lr']
        for paramgroup in paramgroup_options:
            paramgroup['lr'] = base_lr * paramgroup.pop('lr_mult')
            paramgroup['params'] = getattr(model, paramgroup['params']).parameters()
            param_list.append(paramgroup)
        return obj_from_dict(optimizer_cfg, torch.optim,
                             dict(params=param_list))

    paramwise_options = optimizer_cfg.pop('paramwise_options', None)
    # if no paramwise option is specified, just use the global setting
    if paramwise_options is None:
        return obj_from_dict(optimizer_cfg, torch.optim,
                             dict(params=model.parameters()))
    else:
        assert isinstance(paramwise_options, dict)
        # get base lr and weight decay
        base_lr = optimizer_cfg['lr']
        base_wd = optimizer_cfg.get('weight_decay', None)
        # weight_decay must be explicitly specified if mult is specified
        if ('bias_decay_mult' in paramwise_options
                or 'norm_decay_mult' in paramwise_options):
            assert base_wd is not None
        # get param-wise options
        bias_lr_mult = paramwise_options.get('bias_lr_mult', 1.)
        bias_decay_mult = paramwise_options.get('bias_decay_mult', 1.)
        norm_decay_mult = paramwise_options.get('norm_decay_mult', 1.)
        # set param-wise lr and weight decay
        params = []
        for name, param in model.named_parameters():
            param_group = {'params': [param]}
            if not param.requires_grad:
                # FP16 training needs to copy gradient/weight between master
                # weight copy and model weight, it is convenient to keep all
                # parameters here to align with model.parameters()
                params.append(param_group)
                continue

            # for norm layers, overwrite the weight decay of weight and bias
            # TODO: obtain the norm layer prefixes dynamically
            if re.search(r'(bn|gn)(\d+)?.(weight|bias)', name):
                if base_wd is not None:
                    param_group['weight_decay'] = base_wd * norm_decay_mult
            # for other layers, overwrite both lr and weight decay of bias
            elif name.endswith('.bias'):
                param_group['lr'] = base_lr * bias_lr_mult
                if base_wd is not None:
                    param_group['weight_decay'] = base_wd * bias_decay_mult
            # otherwise use the global settings

            params.append(param_group)

        optimizer_cls = getattr(torch.optim, optimizer_cfg.pop('type'))
        return optimizer_cls(params, **optimizer_cfg)


def visualize_effective_dilation(dataloader, model, optimizer, cfg):
    for data in dataloader:
        optimizer.zero_grad()
        data.update(cfg.data.extra)
        model.eval()
        ori_offset = np.array([-1, -1, -1, 0, -1, 1, 0, -1, 0, 0, 0, 1, 1, -1, 1, 0, 1, 1]) * 2
        # data['img'], data['HHA'] = data['img']._data[0].cuda(), data['HHA']._data[0].cuda()
        # feat_c4 = model.module.extract_feat([data['img'], data['HHA']])
        data['img'] = data['img']._data[0].cuda()
        feat_c4 = model.module.extract_feat(data['img'])

        offset, mask = model.module.backbone.layer4[0].conv2.offset.round().detach().cpu().numpy()[0], \
                       model.module.backbone.layer4[0].conv2.mask.detach().cpu().numpy()[0]
        feat_h, feat_w = offset.shape[1:]
        label = cv2.resize(data['gt_label']._data[0][0].numpy(), (feat_w, feat_h), interpolation=cv2.INTER_NEAREST)
        equal, nequal = [], []
        for j in range(feat_h):
            for k in range(feat_w):
                cur_label = label[j, k]
                for l in range(9):
                    sam_h, sam_w = j + ori_offset[2 * l] + offset[2 * l, j, k], k + ori_offset[2 * l + 1] + offset[
                        2 * l + 1, j, k]
                    if sam_h < 0 or sam_h >= feat_h or sam_w < 0 or sam_w >= feat_w:
                        continue
                    samp_label = label[int(sam_h), int(sam_w)]
                    if samp_label == cur_label:
                        equal.append(mask[l, j, k])
                    else:
                        nequal.append(mask[l, j, k])
    print(np.mean(equal), np.std(equal), np.mean(nequal), np.std(nequal))


def parse_args():
    parser = argparse.ArgumentParser(description='DetSeg find lr')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--gpus',
        type=int,
        default=1,
        help='number of gpus to use '
             '(only applicable to non-distributed training)')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument(
        '--autoscale-lr',
        action='store_true',
        help='automatically scale lr with the number of gpus')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()

    cfg = mmcv.Config.fromfile(args.config)
    cfg.data.imgs_per_gpu = 1
    cfg.model.pretrained = None
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    if args.autoscale_lr:
        # apply the linear scaling rule (https://arxiv.org/abs/1706.02677)
        cfg.optimizer['lr'] = cfg.optimizer['lr'] * cfg.gpus / 8

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    logger = get_root_logger(cfg.log_level)
    logger.info('Distributed training: {}'.format(distributed))

    dataset = build_dataset(cfg.data.val)
    data_loader = build_dataloader(
        dataset,
        imgs_per_gpu=cfg.data.imgs_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed)

    if cfg.model_type == 'segmentation':
        model = build_segmentor(cfg.model)
    elif cfg.model_type == 'detection':
        model = build_detector(
            cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
    else:
        raise KeyError('Unrecognized model type: {}'.format(cfg.model_type))

    load_checkpoint(model, args.checkpoint, strict=True, map_location='cpu')

    # put model on gpus
    if args.launcher == 'none':
        model = MMDataParallel(model, device_ids=range(1)).cuda()
    else:
        model = MMDistributedDataParallel(model.cuda())

    # build runner
    optimizer = build_optimizer(model, cfg.optimizer)

    visualize_effective_dilation(data_loader, model, optimizer, cfg)


if __name__ == '__main__':
    main()
