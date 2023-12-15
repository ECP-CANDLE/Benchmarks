from __future__ import absolute_import

__author__ = "Todd Young"
__email__ = "youngmt1@ornl.gov"
__version__ = "0.1.0"

from .api.config import banner

# Essential pieces
from .architecture import Architecture

# Utilities that are not neccessary
from .datasets.p3b3 import P3B3
from .datasets.random import RandomData
from .datasets.sample import sample
from .datasets.uno import Uno
from .functional import multitask_accuracy, multitask_accuracy_topk, multitask_loss
from .meters.accuracy import MultitaskAccuracyMeter
from .meters.average import AverageMeter
from .meters.epoch import EpochMeter
from .modules.conv.network import ConvNetwork
from .modules.linear.network import LinearNetwork
from .modules.network import Network
from .storage.genotype import GenotypeStorage
from .utils.random import SeedControl
from .utils.tensor import to_device

__all__ = [
    "Architecture",
    "Network",
    "ConvNetwork",
    "LinearNetwork",
]
