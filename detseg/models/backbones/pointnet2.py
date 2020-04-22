import math

import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm

from ..registry import BACKBONES
from ..utils import ConvModule
from detseg.ops import QueryAndGroup, GroupAll, three_interpolate, three_nn, gather_operation, furthest_point_sample

@BACKBONES.register_module
class PointNet2(nn.Module):
    def __init__(self,
                 in_channels=6,
                 use_xyz=True, 
                 use_decoder=False,
                 use_HHA=True,
                 norm_cfg=dict(type='BN', requires_grad=True)):
        super(PointNet2, self).__init__()
        self.use_decoder = use_decoder
        self.use_HHA = use_HHA
        
        c_in = in_channels
        self.layer1 = SA(
            npoint=1024,
            radii=[0.05, 0.1],
            nsamples=[16, 32],
            mlps=[[c_in, 16, 16, 32], [c_in, 32, 32, 64]],
            use_xyz=use_xyz,
            use_HHA=2 if use_HHA else -1,
            norm_cfg=norm_cfg)
        c_out_0 = 32 + 64

        c_in = c_out_0
        self.layer2 = SA(
            npoint=256,
            radii=[0.1, 0.2],
            nsamples=[16, 32],
            mlps=[[c_in, 64, 64, 128], [c_in, 64, 96, 128]],
            use_xyz=use_xyz,
            use_HHA=1 if use_HHA else -1,
            norm_cfg=norm_cfg)
        c_out_1 = 128 + 128

        c_in = c_out_1
        self.layer3 = SA(
            npoint=64,
            radii=[0.2, 0.4],
            nsamples=[16, 32],
            mlps=[[c_in, 128, 196, 256], [c_in, 128, 196, 256]],
            use_xyz=use_xyz,
            use_HHA=0 if use_HHA else -1,
            norm_cfg=norm_cfg)
        c_out_2 = 256 + 256

        c_in = c_out_2
        self.layer4 = SA(
            npoint=16,
            radii=[0.4, 0.8],
            nsamples=[16, 32],
            mlps=[[c_in, 256, 256, 512], [c_in, 256, 384, 512]],
            use_xyz=use_xyz,
            use_HHA=0 if use_HHA else -1,
            norm_cfg=norm_cfg)
        c_out_3 = 512 + 512

        if self.use_decoder: 
            self.up_layer1 = FP(mlp_param=[256 + 6, 128, 128], norm_cfg=norm_cfg)
            self.up_layer2 = FP(mlp_param=[512 + c_out_0, 256, 256], norm_cfg=norm_cfg)
            self.up_layer3 = FP(mlp_param=[512 + c_out_1, 512, 512], norm_cfg=norm_cfg)
            self.up_layer4 = FP(mlp_param=[c_out_3 + c_out_2, 512, 512], norm_cfg=norm_cfg)

            self.fc_lyaer = nn.Sequential(
                nn.Conv1d(128, 128, kernel_size=1, bias=False),
                nn.BatchNorm1d(128),
                nn.ReLU(True),
                nn.Dropout(0.5),
                nn.Conv1d(128, 13, kernel_size=1),
            )
    
    def init_weights(self, pretrained=None):
        # ignore args "pretrained" here
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            
    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None
        return xyz, features
    
    def forward(self, pointcloud): 
        if self.use_HHA: 
            xyz, features = pointcloud 
        else:
            xyz, features = self._break_up_pc(pointcloud)

        l_xyz, l_features = [xyz], [features]
        for i in range(4):
            layer = getattr(self, 'layer{}'.format(i+1))
            li_xyz, li_features = layer(l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)

        if self.use_decoder: 
            for i in range(-1, -5, -1):
                up_layer = getattr(self, 'up_layer{i}'.format(5+i))
                l_features[i - 1] = up_layer(
                    l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i]
                )
            out = self.fc_lyaer(l_features[0])
        
        #TODO: return out when needed
        return tuple(l_features)

def build_shared_mlp(mlp_param, norm_cfg):
    layers = [] 
    for i in range(1, len(mlp_param)): 
        layers.append(
            ConvModule(mlp_param[i-1],
                       mlp_param[i],
                       kernel_size=1,
                       norm_cfg=norm_cfg))
    return nn.Sequential(*layers)        

class SA(nn.Module):
    def __init__(self,
               npoint, 
               radii,
               nsamples, 
               mlps,
               use_xyz=True, 
               use_HHA=0,
               norm_cfg=dict(type='BN', requires_grad=True)):
        super(SA, self).__init__()
        assert len(radii) == len(nsamples) == len(mlps)
        self.use_HHA = use_HHA
        self.groupers, self.mlps = [], []
        
        for i in range(len(radii)): 
            grouper = QueryAndGroup(radii[i], nsamples[i], use_xyz=use_xyz) if npoint is not None else GroupAll(use_xyz)
            grouper_name = 'grouper{}'.format(i+1)
            self.add_module(grouper_name, grouper)
            self.groupers.append(grouper_name)
            
            mlp_param = mlps[i] 
            if use_xyz: mlp_param[0] += 3 
            mlp = build_shared_mlp(mlp_param, norm_cfg)
            mlp_name = 'mlp{}'.format(i+1)
            self.add_module(mlp_name, mlp)
            self.mlps.append(mlp_name)
    
    def forward(self, xyz, features): 
        new_features_list = []
        
        if self.use_HHA != -1: 
            new_xyz = xyz
            for i in range(self.use_HHA): 
                new_xyz = F.max_pool2d(new_xyz, kernel_size=1, stride=2)
            B, C, _, _ = features.size()
            _, _, H, W = new_xyz.size()
            xyz = xyz.view(B, 3, -1).transpose(1, 2).contiguous()
            new_xyz = new_xyz.view(B, 3, -1).transpose(1, 2).contiguous()
            features = features.view(B, C, -1)
        else:
            xyz_flipped = xyz.transpose(1, 2).contiguous()
            new_xyz = (gather_operation(xyz_flipped, furthest_point_sample(xyz, self.npoint))
                .transpose(1, 2)
                .contiguous()
                if self.npoint is not None
                else None)
        
        for i in range(len(self.groupers)): 
            grouper = getattr(self, self.groupers[i])
            mlp = getattr(self, self.mlps[i])
            
            new_features = grouper(xyz, new_xyz, features) # (B, C, npoint, nsample)

            new_features = mlp(new_features)  
            new_features = F.max_pool2d(new_features, kernel_size=[1, new_features.size(3)])  
            new_features = new_features.squeeze(-1) 

            new_features_list.append(new_features)

        return new_xyz.transpose(1, 2).view(B, 3, H, W).contiguous(), torch.cat(new_features_list, dim=1).view(B, -1, H, W)

class FP(nn.Module):
    def __init__(self, mlp_param, norm_cfg):
        super(FP, self).__init__()
        self.mlp = build_shared_mlp(mlp_param, bn=bn)

    def forward(self, unknown, known, unknow_feats, known_feats):
        if known is not None:
            dist, idx = three_nn(unknown, known)
            dist_recip = 1.0 / (dist + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm

            interpolated_feats = three_interpolate(
                known_feats, idx, weight
            )
        else:
            interpolated_feats = known_feats.expand(
                *(known_feats.size()[0:2] + [unknown.size(1)])
            )

        if unknow_feats is not None:
            new_features = torch.cat(
                [interpolated_feats, unknow_feats], dim=1
            )  # (B, C2 + C1, n)
        else:
            new_features = interpolated_feats

        new_features = new_features.unsqueeze(-1)
        new_features = self.mlp(new_features)

        return new_features.squeeze(-1)
