import argparse
import os
import re
import cv2
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as mcol
import matplotlib.pyplot as plt

import mmcv
import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import load_checkpoint, obj_from_dict

from detseg.apis import init_dist, get_root_logger, build_optimizer
from detseg.datasets import build_dataloader, build_dataset
from detseg.models import build_segmentor, build_detector


def get_bottom_points(offsets, top_points):
    plot_level = 3
    for i in range(1, plot_level+1):
        offset = offsets[-i]
        map_h, map_w = offset.size()[2:]
        if i == 3:
            vis_attr = {'dilation': 2, 'pad': 2, 'filter_size': 3}
        else:
            vis_attr = {'dilation': 4, 'pad': 4, 'filter_size': 3}
        source_points = []
        for idx, cur_top_point in enumerate(top_points):
            cur_top_point = np.round(cur_top_point)
            if cur_top_point[0] < 0 or cur_top_point[1] < 0 \
                or cur_top_point[0] > map_h-1 or cur_top_point[1] > map_w-1:
                continue
            cur_source_point = kernel_inv_map(vis_attr, cur_top_point, map_h, map_w)
            cur_offset = np.squeeze(offset[:, :, int(cur_top_point[0]), int(cur_top_point[1])])
            cur_source_point = offset_inv_map(cur_source_point, cur_offset)
            source_points = source_points + cur_source_point
        top_points = source_points
    return top_points


def kernel_inv_map(vis_attr, target_point, map_h, map_w):
    pos_shift = [vis_attr['dilation'] * 0 - vis_attr['pad'],
                 vis_attr['dilation'] * 1 - vis_attr['pad'],
                 vis_attr['dilation'] * 2 - vis_attr['pad']]
    source_point = []
    for idx in range(vis_attr['filter_size']**2):
        cur_source_point = np.array([target_point[0] + pos_shift[idx // 3],
                                     target_point[1] + pos_shift[idx % 3]])
        if cur_source_point[0] < 0 or cur_source_point[1] < 0 \
                or cur_source_point[0] > map_h - 1 or cur_source_point[1] > map_w - 1:
            continue
        source_point.append(cur_source_point.astype('f'))
    return source_point


def offset_inv_map(source_points, offset):
    for idx, _ in enumerate(source_points):
        source_points[idx][0] += offset[2*idx]
        source_points[idx][1] += offset[2*idx + 1]
    return source_points

def show_erf_esl(image, points, feat, source_point):
    # create fake loss
    image_h, image_w = image.size()[2:]
    feat_h, feat_w = feat.size()[2:]
    feat_mask = torch.zeros_like(feat)
    feat_mask[..., source_point[0], source_point[1]] = 1.
    fm_mask = feat_mask.cuda()
    fake_loss = torch.mean(feat * feat_mask)
    fake_loss.backward(retain_graph=True)

    # visualize input grad
    image_grad = image.grad.detach().cpu().numpy()[0]
    image_grad = np.abs(np.mean(image_grad, axis=0))

    # magic number: 0.0454
    grad_thr = 0.0454 * np.max(image_grad)
    # grad_thr = 1 * image_grad[int(np.round((source_point[0] + 0.5) * image_h / feat_h)), int(np.round((source_point[1] + 0.5) * image_w / feat_w))]
    mask_grad = image_grad > grad_thr
    mask_grad = mask_grad.astype(np.bool)

    image = image[0].permute(1,2,0).detach().cpu().numpy() + [123.675, 116.28, 103.53]

    color_image = image.copy()
    color_grad = image_grad.copy()

    # normalize to (0, 1)
    min_, max_ = float('inf'), -float('inf')
    for point in points:
        y = np.round((point[0] + 0.5) * image_h / feat_h).astype('i')
        x = np.round((point[1] + 0.5) * image_w / feat_w).astype('i')

        if x < 0 or y < 0 or x > image_w - 1 or y > image_h - 1:
            continue
        min_ = min(min_, color_grad[y, x])
        max_ = max(max_, color_grad[y, x])
    color_grad[color_grad < min_] = min_
    color_grad[color_grad > max_] = max_
    color_grad = (color_grad - min_) / (max_ - min_)

    color_grad[color_grad < 0.01] = 0.01
    color_grad = np.log10(color_grad)
    color_grad = (color_grad + 2) / 2

    # Make a user-defined colormap.
    cm1 = mcol.LinearSegmentedColormap.from_list("MyCmapName", ["b", "r"])
    # Make a normalizer that will map the time values from
    # [start_time,end_time+1] -> [0,1].
    cnorm = mcol.Normalize(vmin=0, vmax=1)
    # Turn these into an object that can be used to map time values to colors and
    # can be passed to plt.colorbar().
    cpick = cm.ScalarMappable(norm=cnorm, cmap=cm1)
    cpick.set_array([])

    image[:, :, 0][mask_grad] = 255.
    image[:, :, 1][mask_grad] = 0.
    image[:, :, 2][mask_grad] = 0.

    for point in points:
        y = np.round((point[0] + 0.5) * image_h / feat_h).astype('i')
        x = np.round((point[1] + 0.5) * image_w / feat_w).astype('i')

        if x < 0 or y < 0 or x > image_w - 1 or y > image_h - 1:
            continue
        color = (np.array(cpick.to_rgba(color_grad[y, x])[:3]) * 255).astype(np.uint8)
        y = min(y, image_h - 2 - 1)
        x = min(x, image_w - 2 - 1)
        y = max(y, 2)
        x = max(x, 2)
        color_image[y - 2:y + 2 + 1, x - 2:x + 2 + 1, :] = np.tile(
            np.reshape(color, (1, 1, 3)), (2 * 2 + 1, 2 * 2 + 1, 1)
        )

    mapw_sp, maph_sp = int(np.round((source_point[1] + 0.5) * image_w / feat_w)), int(np.round((source_point[0] + 0.5) * image_h / feat_h))
    image[maph_sp - 2: maph_sp +3, mapw_sp - 2:mapw_sp + 3, :] = np.tile(
            np.reshape(np.array([0, 255, 0]), (1, 1, 3)), (2 * 2 + 1, 2 * 2 + 1, 1)
        )
    color_image[maph_sp - 2: maph_sp + 3, mapw_sp - 2:mapw_sp + 3, :] = np.tile(
        np.reshape(np.array([0, 255, 0]), (1, 1, 3)), (2 * 2 + 1, 2 * 2 + 1, 1)
    )
    cv2.imwrite('image.png', image[:, :, ::-1])
    cv2.imwrite('color_image.png', color_image[:, :, ::-1])

    fig = plt.figure()
    ax = fig.add_subplot(121)
    plt.imshow(image / 255)
    ax2 = fig.add_subplot(122)
    plt.imshow(color_image / 255, cmap="jet")
    plt.show()


def visualize_spatial_support(dataloader, model, optimizer, cfg):
    for i, data in enumerate(dataloader):
        if i == 1:
            optimizer.zero_grad()
            data.update(cfg.data.extra)
            model.train()
            # data['img'], data['HHA'] = data['img']._data[0].cuda(), data['HHA']._data[0].cuda()
            data['img'] = data['img']._data[0].cuda()
            data['img'].requires_grad = True
            feat_c4 = model.module.extract_feat(data['img'])
            feat_h, feat_w = feat_c4.size()[2:]
            for cur_h in range(10, feat_h, 10):
                for cur_w in range(10, feat_w, 10):
                    if cur_h == 10 and cur_w == 20:
                        offsets, source_point = [], np.array([cur_h, cur_w])
                        # for i in range(model.module.backbone.stage_blocks[-1]):
                        #     offsets.append(model.module.backbone.layer4[i].conv2.offset)
                        offsets = [torch.zeros((1, 18, 60, 80)).cuda()] * 3
                        points = get_bottom_points(offsets, [source_point])
                        show_erf_esl(data['img'], points, feat_c4, source_point)


def parse_args():
    parser = argparse.ArgumentParser(description='DetSeg spatial support')
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
        model = MMDataParallel(model, device_ids=range(cfg.gpus)).cuda()
    else:
        model = MMDistributedDataParallel(model.cuda())

    # build runner
    optimizer = build_optimizer(model, cfg.optimizer)

    visualize_spatial_support(data_loader, model, optimizer, cfg)


if __name__ == '__main__':
    main()