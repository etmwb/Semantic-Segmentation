import h5py
from PIL import Image
import numpy as np

from ..registry import SEG_PIPELINES

@SEG_PIPELINES.register_module
class LoadImageFromFile(object):

    def __init__(self, label_minus=True):
        self.label_minus = label_minus

    def __call__(self, results):  
        for key, value in results.items():
            if key == 'PC': 
                h5f = h5py.File(value, 'r')
                results[key] = h5f['pcs'][:]
                continue
            results[key] = np.array(Image.open(value))

        # for depth-aware conv
        results['depth'] = results['depth'] / 120.

        if self.label_minus:
            label = results['label'] - 1 
            label[label == -1] = 255 
            results['label'] = label
        return results
