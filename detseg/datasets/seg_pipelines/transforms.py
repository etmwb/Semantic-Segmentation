import mmcv
import cv2
import numpy as np
import matplotlib as mpl

from ..registry import SEG_PIPELINES

@SEG_PIPELINES.register_module
class Resize(object):
    def __init__(self,
                 scale,
                 shorter_side):
        self.scale = np.random.uniform(min(scale), max(scale))
        self.shorter_side = shorter_side

    def __call__(self, results):
        image_h, image_w = results['label'].shape
        results['ori_shape'] = (image_h, image_w)
        if self.shorter_side == 'adaptive': 
            self.shorter_side = int(image_h * 0.73)
        short_side = min(image_h, image_w)
        if short_side * self.scale < self.shorter_side:
            self.scale = self.shorter_side * 1. / short_side
        results['img'] = cv2.resize(results['img'], None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_CUBIC)
        results['label'] = cv2.resize(results['label'], None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_NEAREST)
        if 'depth' in results:
            results['depth'] = results['depth'].astype(np.float32)
            results['depth'] = cv2.resize(results['depth'], None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_CUBIC)
        if 'HHA' in results:
            results['HHA'] = cv2.resize(results['HHA'], None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_CUBIC)
        if 'PC' in results: 
            results['PC'] = cv2.resize(results['PC'], None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_CUBIC)

        return results

@SEG_PIPELINES.register_module
class PadCrop(object):
    def __init__(self,
                 pad_value,
                 crop_size):
        self.pad_value = pad_value
        self.crop_size = crop_size

    def __call__(self, results):
        img, label = results['img'], results['label']
        ori_shape = results.pop('ori_shape')
        if self.crop_size == 'adaptive': 
            size = min(ori_shape[0], 520)
            self.crop_size = (size, size)

        h, w, c = img.shape
        ch, cw = self.crop_size
        expand_h, expand_w = max(h, ch), max(w, cw)
        left = max(0, expand_w - w) // 2
        top = max(0, expand_h - h) // 2
        crop_left = np.random.randint(0, expand_w - cw + 1)
        crop_top = np.random.randint(0, expand_h - ch + 1)

        expand_img = np.full((expand_h, expand_w, c),
                             self.pad_value['img']).astype(img.dtype)
        expand_img[top:top + h, left:left + w] = img
        expand_img = expand_img[crop_top:crop_top + ch, crop_left:crop_left + cw]
        results['img'] = expand_img

        expand_label = np.full((expand_h, expand_w),
                               self.pad_value['label']).astype(label.dtype)
        expand_label[top:top + h, left:left + w] = label
        expand_label = expand_label[crop_top:crop_top + ch, crop_left:crop_left + cw]
        results['label'] = expand_label

        if 'depth' in results:
            expand_depth = np.full((expand_h, expand_w),
                               self.pad_value['depth']).astype(results['depth'].dtype)
            expand_depth[top:top + h, left:left + w] = results['depth']
            expand_depth = expand_depth[crop_top:crop_top + ch, crop_left:crop_left + cw]
            results['depth'] = expand_depth

        if 'HHA' in results:
            expand_HHA = np.full((expand_h, expand_w, c),
                               self.pad_value['HHA']).astype(results['HHA'].dtype)
            expand_HHA[top:top + h, left:left + w] = results['HHA']
            expand_HHA = expand_HHA[crop_top:crop_top + ch, crop_left:crop_left + cw]
            results['HHA'] = expand_HHA
        
        if 'PC' in results:
            expand_PC = np.full((expand_h, expand_w, c),
                               self.pad_value['PC']).astype(results['PC'].dtype)
            expand_PC[top:top + h, left:left + w] = results['PC']
            expand_PC = expand_PC[crop_top:crop_top + ch, crop_left:crop_left + cw]
            results['PC'] = expand_PC

        return results

@SEG_PIPELINES.register_module
class RandomFlip(object):

    def __init__(self, flip_ratio=0.5):
        self.flip_ratio = flip_ratio
        assert self.flip_ratio >= 0 and flip_ratio <= 1

    def __call__(self, results):
        if np.random.rand() < self.flip_ratio:
            results['img'] = mmcv.imflip(results['img'])
            results['label'] = mmcv.imflip(results['label'])
            if 'depth' in results:
                results['depth'] = mmcv.imflip(results['depth'])
            if 'HHA' in results:
                results['HHA'] = mmcv.imflip(results['HHA'])
            if 'PC' in results:
                results['PC'] = mmcv.imflip(results['PC'])

        return results

@SEG_PIPELINES.register_module
class RandomHSV(object):

    def __init__(self, h_scale,
                 s_scale,
                 v_scale):
        assert isinstance(h_scale, (list, tuple)) and \
               isinstance(s_scale, (list, tuple)) and \
               isinstance(v_scale, (list, tuple))
        self.h_scale = h_scale
        self.s_scale = s_scale
        self.v_scale = v_scale

    def __call__(self, results):
        img = results['img']
        img_hsv = mpl.colors.rgb_to_hsv(img)
        img_h, img_s, img_v = img_hsv[:, :, 0], img_hsv[:, :, 1], img_hsv[:, :, 2]
        h_random = np.random.uniform(min(self.h_scale), max(self.h_scale))
        s_random = np.random.uniform(min(self.s_scale), max(self.s_scale))
        v_random = np.random.uniform(-min(self.v_scale), max(self.v_scale))
        img_h = np.clip(img_h * h_random, 0, 1)
        img_s = np.clip(img_s * s_random, 0, 1)
        img_v = np.clip(img_v + v_random, 0, 255)
        img_hsv = np.stack([img_h, img_s, img_v], axis=2)
        results['img'] = mpl.colors.hsv_to_rgb(img_hsv)

        return results

@SEG_PIPELINES.register_module
class Normalize(object):

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, results):
        img_mean = np.array(self.mean['img'], dtype=np.float32)
        img_std = np.array(self.std['img'], dtype=np.float32)
        results['img'] = mmcv.imnormalize(results['img'], img_mean, img_std, False)

        if 'HHA' in results:
            HHA_mean = np.array(self.mean['HHA'], dtype=np.float32)
            HHA_std = np.array(self.std['HHA'], dtype=np.float32)
            results['HHA'] = mmcv.imnormalize(results['HHA'], HHA_mean, HHA_std, False)

        return results
