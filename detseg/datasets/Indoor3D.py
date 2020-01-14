import torch
import numpy as np
import os.path as osp
import h5py

from .base import BaseDataset


class Indoor3D(BaseDataset):
    def __init__(self,
                 pipeline,
                 data_root,
                 num_points,
                 data_percent=1.0,
                 test_mode=False):
        super(Indoor3D, self).__init__(pipeline, data_root, test_mode=test_mode)
        self.num_points, self.data_percent = num_points, data_percent
        self.load_path()


    def load_path(self):
        with open(osp.join(self.data_root, 'all_files.txt'), 'r') as f:
            all_files = [line.rstrip() for line in f]
        with open(osp.join(self.data_root, 'room_filelist.txt'), 'r') as f:
            room_files = [line.rstrip() for line in f]

        points, labels = [], []
        for file in all_files:
            file_dict = h5py.File(osp.join(self.data_root, file))
            data, label = f['data'][:], f['label'][:]
            points.append(data)
            labels.append(label)

        points = np.concatenate(points, 0)
        labels = np.concatenate(labels, 0)

        train_idxs, test_idxs = [], []
        for i, room_name in enumerate(room_files):
            if 'Area_5' in room_name:
                test_idxs.append(i)
            else:
                train_idxs.append(i)

        if self.test_mode:
            self.points = points[train_idxs, ...]
            self.labels = labels[train_idxs, ...]
        else:
            self.points = points[test_idxs, ...]
            self.labels = labels[test_idxs, ...]


    def __getitem__(self, idx):
        pt_idxs = np.arange(0, self.num_points)
        np.random.shuffle(pt_idxs)

        current_points = torch.from_numpy(self.points[idx, pt_idxs].copy()).type(
            torch.FloatTensor
        )
        current_labels = torch.from_numpy(self.labels[idx, pt_idxs].copy()).type(
            torch.LongTensor
        )

        return current_points, current_labels


    def __len__(self):
        return int(self.points.shape[0] * self.data_precent)


    def set_num_points(self, pts):
        self.num_points = pts
