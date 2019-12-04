import logging
from abc import ABCMeta, abstractmethod

import torch.nn as nn


class BaseSegmentor(nn.Module):
    """Base class for segmentors"""

    __metaclass__ = ABCMeta