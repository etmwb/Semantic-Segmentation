import os.path as osp
import numpy as np

from .base import BaseDataset
from .registry import DATASETS

@DATASETS.register_module
class Nyuv2Dataset(BaseDataset):

    CLASSES = ('wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table',
               'door', 'window', 'bookshelf', 'picture', 'counter', 'blinds',
               'desk', 'shelves', 'curtain', 'dresser', 'pillow', 'mirror',
               'floor mat', 'clothes', 'ceiling', 'books', 'refridgerator',
               'television', 'paper', 'towel', 'shower curtain', 'box',
               'whiteboard', 'person', 'night stand', 'toilet', 'sink',
               'lamp', 'bathtub', 'bag', 'otherstructure', 'otherfurniture',
               'otherprop')

    def __init__(self,
                 path_file,
                 pipeline,
                 data_root=None,
                 img_prefix=None,
                 seg_prefix=None,
                 test_mode=False):
        super(Nyuv2Dataset, self).__init__(pipeline, data_root, test_mode=test_mode)
        self.path_file = path_file
        self.paths = self.load_path()
        self.flag = np.zeros(len(self), dtype=np.uint8)

    def load_path(self):
        image_paths, depth_paths, label_paths, HHA_paths = [], [], [], []
        with open(osp.join(self.data_root, self.path_file), 'r') as f:
            for postfix in f.readlines():
                postfix = postfix.strip()
                image_paths.append(osp.join(self.data_root, 'image', postfix))
                depth_paths.append(osp.join(self.data_root, 'depth', postfix))
                label_paths.append(osp.join(self.data_root, 'label', postfix))
                HHA_paths.append(osp.join(self.data_root, 'HHA', postfix))
        return dict(image_paths=image_paths, depth_paths=depth_paths,
                    label_paths=label_paths, HHA_paths=HHA_paths)

    def __getitem__(self, index):
        result = dict(
            img=self.paths['image_paths'][index], depth=self.paths['depth_paths'][index],
            label=self.paths['label_paths'][index], HHA=self.paths['HHA_paths'][index])
        return self.pipeline(result)

    def __len__(self):
        return len(self.paths['image_paths'])
