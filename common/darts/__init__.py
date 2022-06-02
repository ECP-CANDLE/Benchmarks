from __future__ import absolute_import

__author__ = 'Todd Young'
__email__ = 'youngmt1@ornl.gov'
__version__ = '0.1.0'

# Essential pieces
from .architecture import Architecture
from .modules.network import Network
from .modules.conv.network import ConvNetwork
from .modules.linear.network import LinearNetwork
from .storage.genotype import GenotypeStorage

# Utilities that are not neccessary
from .datasets.p3b3 import P3B3
from .datasets.uno import Uno
from .datasets.random import RandomData
from .datasets.sample import sample
from .api.config import banner
from .meters.average import AverageMeter
from .meters.accuracy import MultitaskAccuracyMeter
from .meters.epoch import EpochMeter
from .utils.tensor import to_device
from .utils.random import SeedControl

from .functional import (
    multitask_loss, multitask_accuracy, multitask_accuracy_topk
)

__all__ = [
    "Architecture",
    "Network",
    "ConvNetwork",
    "LinearNetwork",
]
