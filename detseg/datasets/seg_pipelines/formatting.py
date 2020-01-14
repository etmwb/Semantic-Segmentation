from collections.abc import Sequence

import mmcv
import torch
import numpy as np
from mmcv.parallel import DataContainer as DC

from ..registry import SEG_PIPELINES


def to_tensor(data):
    """Convert objects of various python types to :obj:`torch.Tensor`.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` and :class:`float`.
    """
    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, Sequence) and not mmcv.is_str(data):
        return torch.tensor(data)
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    else:
        raise TypeError('type {} cannot be converted to tensor.'.format(
            type(data)))


@SEG_PIPELINES.register_module
class DefaultFormatBundle(object):

    def __call__(self, results):
        img = np.ascontiguousarray(results['img'].transpose(2, 0, 1))
        results['img'] = DC(to_tensor(img), stack=True)
        label = np.ascontiguousarray(results['label'])
        results['label'] = DC(to_tensor(label).long(), stack=True, padding_value=255)
        if 'depth' in results:
            depth = np.ascontiguousarray(np.expand_dims(results['depth'], axis=0))
            results['depth'] = DC(to_tensor(depth), stack=True)
        if 'HHA' in results:
            HHA = np.ascontiguousarray(results['HHA'].transpose(2, 0, 1))
            results['HHA'] = DC(to_tensor(HHA), stack=True)

        return results


@SEG_PIPELINES.register_module
class Collect(object):
    def __init__(self, keys=[]):
        self.keys = keys

    def __call__(self, results):
        data = {}
        data['img'] = results['img']
        data['gt_label'] = results['label']
        for key in self.keys:
            data[key] = results[key]

        return data
