import math 

import torch.nn as nn 

from ..registry import BACKBONES
from ..utils import ConvModule
from detseg.ops import QueryAndGroup, GroupAll, gather_operation, furthest_point_sample

@BACKBONES.register_module
class PointNet2(nn.Module):
    def __init__(self,
                 use_xyz=True, 
                 use_decoder=False,
                 norm_cfg=dict(type='BN', requires_grad=True)):
        super(PointNet2, self).__init__()


class SA(nn.Module):
    def __init(self,
               npoint, 
               radii,
               nsample, 
               mlps,
               use_xyz=True, 
               norm_cfg=dict(type='BN', requires_grad=True)):
        super(SA, self).__init__()
        assert len(radii) == len(nsample) == len(mlps)
        self.groupers, self.mlps = [], []
        
        for i in range(len(radii)): 
            grouper = QueryAndGroup(radii[i], nsample[i], use_xyz=use_xyz) if npoint is not None
                    else GroupAll(use_xyz)
            grouper_name = 'grouper{}'.format(i+1)
            self.add_module(grouper_name, grouper)
            self.groupers.append(grouper_name)
            
            mlp_param = mlps[i] 
            if use_xyz: mlp_param[0] += 3 
            mlp = self.build_shared_mlp(mlp_param, norm_cfg)
            mlp_name = 'mlp{}'.format(i+1)
            self.add_module(mlp_name, mlp)
            self.mlps.append(mlp_name)
                
    def build_shared_mlp(self, mlp_param, norm_cfg):
        layers = [] 
        for i in range(1, len(mlp_param)): 
            layers.append(
                ConvModule(mlp_param[i-1],
                           mlp_param[i],
                           kernel_size=1,
                           norm_cfg=norm_cfg))
        return nn.Sequential(*layers)
    
    def forward(self, xyz, features): 
        new_features_list = []
        
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

        return new_xyz, torch.cat(new_features_list, dim=1)
