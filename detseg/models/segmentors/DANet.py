from ..registry import SEGMENTORS
from .single_stage import SingleStageSegmentor

@SEGMENTORS.register_module
class DANet(SingleStageSegmentor):

    def __init__(self,
                 backbone,
                 head,
                 backbone_depth=None,
                 pretrained=None):
        super(DANet, self).__init__(
            backbone=backbone,
            head=head,
            backbone_depth=backbone_depth,
            pretrained=pretrained
        )

    def extract_feat_hybrid(self, img):
        img, depth = img

        depth = self.backbone_depth(depth)
        x = self.backbone.conv1(img)
        x = self.backbone.norm1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        for i, layer_name in enumerate(self.backbone.res_layers):
            res_layer = getattr(self.backbone, layer_name)
            if i == 0:
                x = res_layer(x)
            elif i == 1:
                for j in range(self.backbone.stage_blocks[i]):
                    if j == 0:
                        x, _ = res_layer[j]([x, depth[i-1]])
                    else:
                        x, _ = res_layer[j]([x, depth[i]])
            else:
                x, _ = res_layer([x, depth[i]])

        return x
