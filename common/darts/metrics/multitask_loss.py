import torch.nn as nn

import darts.functional as F
from darts.api.metrics.average import MultitaskAveragingSupervisedMetric


class MultitaskLoss(MultitaskAveragingSupervisedMetric):
    """ Multitask Classification loss """

    def __init__(self, scope="train", criterion=nn.CrossEntropyLoss()):
        super().__init__("loss", scope=scope)
        self.criterion = criterion

    def _value_function(self, x_input, y_true, y_pred, reduce=None):
        """ Return loss value of input """
        return F.multitask_loss(y_true, y_pred, criterion=self.criterion, reduce=reduce)


def create():
    """ darts factory function """
    return MultitaskLoss()
