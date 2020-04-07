import torch
import torch.nn as nn
from torch.autograd import Function 

from . import set_abstraction_cuda

class FurthestPointSampling(Function):
    @staticmethod
    def forward(ctx, xyz, npoints):
        return set_abstraction_cuda.furthest_point_sampling(xyz, npoints)

    @staticmethod
    def backward(ctx, grad_idx):
        return None, None

class GatherOperation(Function):
    @staticmethod
    def forward(ctx, features, idxs):
        N = features.size()[-1]
        ctx.for_backwards = (idxs, N)
        return set_abstraction_cuda.gather_points_forward(features, idxs)

    @staticmethod
    def backward(ctx, grad_out):
        idxs, N = ctx.for_backwards
        grad_features = set_abstraction_cuda.gather_points_backward(grad_out.contiguous(), idxs, N)
        return grad_features, None

class ThreeNN(Function):
    @staticmethod
    def forward(ctx, unknown, known):
        dist2, idxs = set_abstraction_cuda.three_nn(unknown, known)
        return torch.sqrt(dist2), idxs

    @staticmethod
    def backward(ctx, grad_unk, grad_kno):
        return None, None

class ThreeInterpolate(Function):
    @staticmethod
    def forward(ctx, features, idxs, weight):
        B, c, m = features.size()
        ctx.three_interpolate_for_backward = (idxs, weight, m)
        return set_abstraction_cuda.three_interpolate_forward(features, idxs, weight)

    @staticmethod
    def backward(ctx, grad_out):
        idxs, weight, m = ctx.three_interpolate_for_backward
        grad_features = set_abstraction_cuda.three_interpolate_backward(
            grad_out.contiguous(), idxs, weight, m
        )
        return grad_features, None, None

class GroupingOperation(Function):
    @staticmethod
    def forward(ctx, features, idxs):
        N = features.size()[-1]
        ctx.for_backwards = (idxs, N)
        return set_abstraction_cuda.group_points_forward(features, idxs)

    @staticmethod
    def backward(ctx, grad_out):
        idxs, N = ctx.for_backwards
        grad_features = set_abstraction_cuda.group_points_backward(grad_out.contiguous(), idxs, N)
        return grad_features, None

class BallQuery(Function):
    @staticmethod
    def forward(ctx, radius, nsamples, xyz, new_xyz):
        return set_abstraction_cuda.ball_query(new_xyz, xyz, radius, nsamples)

    @staticmethod
    def backward(ctx, grad_idx):
        return None, None, None, None


furthest_point_sample = FurthestPointSampling.apply
gather_operation = GatherOperation.apply
three_nn = ThreeNN.apply
three_interpolate = ThreeInterpolate.apply
grouping_operation = GroupingOperation.apply
ball_query = BallQuery.apply


class QueryAndGroup(nn.Module):
    def __init__(self, radius, nsample, use_xyz=True):
        super(QueryAndGroup, self).__init__()
        self.radius, self.nsample, self.use_xyz = radius, nsample, use_xyz

    def forward(self, xyz, new_xyz, features=None):
        idx = ball_query(self.radius, self.nsample, xyz, new_xyz)
        xyz_trans = xyz.transpose(1, 2).contiguous()
        grouped_xyz = grouping_operation(xyz_trans, idx)  # (B, 3, npoint, nsample)
        grouped_xyz -= new_xyz.transpose(1, 2).unsqueeze(-1)

        if features is not None:
            grouped_features = grouping_operation(features, idx)
            if self.use_xyz:
                new_features = torch.cat(
                    [grouped_xyz, grouped_features], dim=1
                )  # (B, C + 3, npoint, nsample)
            else:
                new_features = grouped_features
        else:
            assert (
                self.use_xyz
            ), "Cannot have not features and not use xyz as a feature!"
            new_features = grouped_xyz

        return new_features


class GroupAll(nn.Module):
    def __init__(self, use_xyz=True):
        super(GroupAll, self).__init__()
        self.use_xyz = use_xyz

    def forward(self, xyz, new_xyz, features=None):
        grouped_xyz = xyz.transpose(1, 2).unsqueeze(2)
        if features is not None:
            grouped_features = features.unsqueeze(2)
            if self.use_xyz:
                new_features = torch.cat(
                    [grouped_xyz, grouped_features], dim=1
                )  # (B, 3 + C, 1, N)
            else:
                new_features = grouped_features
        else:
            new_features = grouped_xyz

        return new_features