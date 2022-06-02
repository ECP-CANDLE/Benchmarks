import torch
import torch.nn as nn

from darts.api import Model
from darts.genotypes import PRIMITIVES
from darts.modules.operations.conv import OPS


class MixedLayer(Model):
    """ A mixture of 8 unit types

    We use weights to aggregate these outputs while training.
    and softmax to select the strongest edges while inference.
    """
    def __init__(self, c, stride):
        super(MixedLayer, self).__init__()
        self.reset(c, stride)

    def reset(self, c, stride):
        self.layers = nn.ModuleList()

        for primitive in PRIMITIVES:
            layer = OPS[primitive](c, stride, False)

            if 'pool' in primitive:
                layer = nn.Sequential(layer, nn.BatchNorm1d(c, affine=False))

            self.layers.append(layer)

    def forward(self, x, weights):
        """
        Parameters
        ----------
        x : torch.tensor
            Data

        Weights : torch.tensor
            alpha, [op_num:8], the output = sum of alpha * op(x)
        """
        x = [w * layer(x) for w, layer in zip(weights, self.layers)]
        return sum(x)
