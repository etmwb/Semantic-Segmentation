from ..registry import SEGMENTORS
from .base import BaseSegmentor

SEGMENTORS.register_module
class DANet(BaseSegmentor):

    def __init__(self,
                 backbones,
                 head,
                 backbones_depth=None,
                 pretrained=None):
