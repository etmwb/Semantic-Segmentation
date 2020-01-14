import argparse
from functools import partial

import torch
from mmcv import Config

from detseg.models import build_detector, build_segmentor
from detseg.utils import get_model_complexity_info


def input_hha(model, shape):
    assert len(shape) == 3
    batch = torch.ones(()).new_empty(
        (1, *shape),
        dtype=next(model.parameters()).dtype,
        device=next(model.parameters()).device)
    return dict(img=batch, gt_label=None, depth=None, HHA=batch)


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[1280, 800],
        help='input image size')
    args = parser.parse_args()
    return args


def main():

    args = parse_args()

    if len(args.shape) == 1:
        input_shape = (3, args.shape[0], args.shape[0])
    elif len(args.shape) == 2:
        input_shape = (3, ) + tuple(args.shape)
    else:
        raise ValueError('invalid input shape')

    cfg = Config.fromfile(args.config)
    if cfg.model_type == 'segmentation':
        model = build_segmentor(cfg.model).cuda()
    elif cfg.model_type == 'detection':
        model = build_detector(cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg).cuda()
    else:
        raise KeyError('Unrecognized model type: {}'.format(cfg.model_type))
    model.eval()

    if hasattr(model, 'forward_dummy'):
        model.forward = model.forward_dummy
    else:
        raise NotImplementedError(
            'FLOPs counter is currently not currently supported with {}'.
            format(model.__class__.__name__))
    # TODO: support DCNv2
    flops, params = get_model_complexity_info(
        model, input_shape, False, input_constructor=partial(input_hha, model) if hasattr(model, 'backbone_depth') else None)
    split_line = '=' * 30
    print('{0}\nInput shape: {1}\nFlops: {2}\nParams: {3}\n{0}'.format(
        split_line, input_shape, flops, params))


if __name__ == '__main__':
    main()
