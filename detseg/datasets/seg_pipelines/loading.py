from PIL import Image
import numpy as np

from ..registry import SEG_PIPELINES

@SEG_PIPELINES.register_module
class LoadImageFromFile(object):

    def __init__(self, to_float32=False):
        self.to_float32 = to_float32

    def __call__(self, results):
        for key, value in results.items():
            results[key] = np.array(Image.open(value))
        return results
