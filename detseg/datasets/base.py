import os.path as osp

from torch.utils.data import Dataset

from .seg_pipelines import Seg_Compose
from .registry import DATASETS


@DATASETS.register_module
class BaseDataset(Dataset):

    CLASSES = None

    def __init__(self,
                 pipeline,
                 data_root=None,
                 img_prefix=None,
                 seg_prefix=None,
                 test_mode=False):
        self.data_root = data_root
        self.img_prefix = img_prefix
        self.seg_prefix = seg_prefix
        self.test_mode = test_mode

        # join paths if data_root is specified
        if self.data_root is not None:
            if not (self.img_prefix is None or osp.isabs(self.img_prefix)):
                self.img_prefix = osp.join(self.data_root, self.img_prefix)
            if not (self.seg_prefix is None or osp.isabs(self.seg_prefix)):
                self.seg_prefix = osp.join(self.data_root, self.seg_prefix)

        self.pipeline = Seg_Compose(pipeline)

    def __getitem__(self, index):
        raise NotImplementedError

    @property
    def num_classes(self):
        return len(self.CLASSES)