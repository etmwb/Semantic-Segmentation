from PIL import Image
import numpy as np

from ..registry import SEG_PIPELINES

@SEG_PIPELINES.register_module
class LoadImageFromFile(object):

    def __init__(self, label_minus=False):
        self.label_minus = label_minus

    def __call__(self, results):
        for key, value in results.items():
            results[key] = np.array(Image.open(value))

        # for depth-aware conv
        results['depth'] = results['depth'] / 120.

        if self.label_minus:
            results['label'] = results['label'] - 1
        return results
