import argparse
import os
import math
import matplotlib.pyplot as plt

import mmcv
import torch
import torch.distributed as dist
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info

from detseg.apis import init_dist, get_root_logger, build_optimizer
from detseg.datasets import build_dataloader, build_dataset
from detseg.models import build_segmentor, build_detector
from detseg.core.utils.dist_utils import allreduce_grads


def adjust_lrs(optimizer, lr):
    base_lr = max([group['lr'] for group in optimizer.param_groups])
    for i, group in enumerate(optimizer.param_groups):
        group['lr'] = group['lr'] / base_lr * lr


def single_gpu_lrfinder(model, data_loader, optimizer, cfg, min_lr=1e-8, max_lr=1., beta=0.98):
    model.train()
    batches_per_gpu = len(data_loader) - 1
    mult = (max_lr / min_lr) ** (1 / batches_per_gpu)

    lr = min_lr
    avg_loss, best_loss, batch_num = 0., 0., 0.
    losses, lrs = [], []

    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for data in data_loader:
        data.update(cfg.data.extra)

        optimizer.zero_grad()
        adjust_lrs(optimizer, lr)

        batch_num += 1
        loss = model(return_loss=True, rescale=False, **data)['loss_head']

        avg_loss = beta * avg_loss + (1 - beta) * loss
        smoothed_loss = avg_loss / (1 - beta ** batch_num)
        # Stop if the loss is exploding
        if batch_num > 1 and smoothed_loss > 4 * best_loss:
            return lrs, losses
        # Record the best loss
        if smoothed_loss < best_loss or batch_num == 1:
            best_loss = smoothed_loss
        # Store the values
        losses.append(smoothed_loss)
        lrs.append(math.log10(lr))
        # Do the SGD step
        loss.backward()
        allreduce_grads(model.parameters())
        optimizer.step()
        # Update the lr for the next step
        lr *= mult

        batch_size = data['img']._data[0].size(0)
        for _ in range(batch_size):
            prog_bar.update()

    plt.plot(lrs[10:-5], losses[10:-5])
    plt.show()


def multi_gpu_lrfinder(model, data_loader, optimizer, cfg, min_lr=1e-8, max_lr=1., beta=0.98):
    model.train()
    batches_per_gpu = len(data_loader) - 1
    mult = (max_lr / min_lr) ** (1 / batches_per_gpu)

    lr = min_lr
    avg_loss, best_loss, batch_num = 0., 0., 0.
    losses, lrs = [], []

    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    for data in data_loader:
        data.update(cfg.data.extra)

        optimizer.zero_grad()
        adjust_lrs(optimizer, lr)

        batch_num += 1
        loss = model(return_loss=True, rescale=False, **data)['loss_head']
        # Do the SGD step
        loss.backward()
        allreduce_grads(model.parameters())
        optimizer.step()

        # sum loss from all gpus
        loss_list = [loss for _ in range(world_size)]
        dist.all_gather(loss_list, loss)
        sum_loss = sum(loss_list)

        if rank == 0:
            avg_loss = beta * avg_loss + (1 - beta) * sum_loss
            smoothed_loss = avg_loss / (1 - beta ** (batch_num * world_size))
            # Stop if the loss is exploding
            if batch_num > 1 and smoothed_loss > 4 * best_loss:
                return lrs, losses
            # Record the best loss
            if smoothed_loss < best_loss or batch_num == 1:
                best_loss = smoothed_loss
            # Store the values
            losses.append(smoothed_loss)
            lrs.append(math.log10(lr))

            batch_size = data['img']._data[0].size(0)
            for _ in range(batch_size * world_size):
                prog_bar.update()

        # Update the lr for the next step
        lr *= mult

    if rank == 0:
        plt.plot(lrs[10:-5], losses[10:-5])
        plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description='DetSeg lrfinder')
    parser.add_argument('config', help='test config file path')
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
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args

def main():
    args = parse_args()

    cfg = mmcv.Config.fromfile(args.config)
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

    dataset = build_dataset(cfg.data.train)
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

    # put model on gpus
    if args.launcher == 'none':
        model = MMDataParallel(model, device_ids=range(cfg.gpus)).cuda()
    else:
        model = MMDistributedDataParallel(model.cuda())

    # build runner
    optimizer = build_optimizer(model, cfg.optimizer)

    if not distributed:
        single_gpu_lrfinder(model, data_loader, optimizer, cfg)
    else:
        multi_gpu_lrfinder(model, data_loader, optimizer, cfg)


if __name__ == '__main__':
    main()