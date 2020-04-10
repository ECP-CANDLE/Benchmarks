from __future__ import absolute_import

__author__ = 'Todd Young'
__email__ = 'youngmt1@ornl.gov'
__version__ = '0.1.0'

# Essential pieces
from .architecture import Architecture
from .modules.conv.network import ConvNetwork
from .modules.linear.network import LinearNetwork
from .storage.genotype import GenotypeStorage

# Utilities that are not neccessary
from .datasets.p3b3 import P3B3
from .api.config import banner
from .meters.average import AverageMeter
from .meters.accuracy import MultitaskAccuracyMeter
from .utils.logging import log_accuracy

from darts.meters.epoch import EpochMeter
from darts.meters.accuracy import MultitaskAccuracyMeter
from darts.utils.tensor import to_device
from darts.utils.random import SeedControl
from darts.utils.logging import log_accuracy
from darts.utils.tensor import to_device

from .functional import (
    multitask_loss, multitask_loss, multitask_accuracy
)

__all__ = [
    "Architecture",
    "ConvNetwork",
    "LinearNetwork",
]

