import darts.functional as F
from darts.api.metrics.average import MultitaskAveragingSupervisedMetric


class MultitaskAccuracy(MultitaskAveragingSupervisedMetric):
    """ Multitask Classification accuracy """

    def __init__(self, scope="train"):
        super().__init__("accuracy", scope=scope)

    def _value_function(self, x_input, y_true, y_pred):
        """ Return classification accuracy of input """
        return F.multitask_accuracy(y_true, y_pred)


def create():
    """ darts factory function """
    return MultitaskAccuracy()
