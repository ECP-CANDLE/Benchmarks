import torch
import torch.nn as nn

from darts.api import Model
from darts.modules.mixed_layer import MixedLayer


class ConvBlock(Model):
    """ ReLu -> Conv2d """

    def __init__(self, c_in, c_out, kernel_size, stride, padding, affine=True):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(
            c_in, c_out, kernel_size=kernel_size, stride=stride, padding=padding
        )

    def forward(self, x):
        return self.conv(x)


class Cell(Model):

    def __init__(self, num_nodes, multiplier, cpp, cp, c, primitives, ops):
        """
        :param steps: 4, number of layers inside a cell
        :param multiplier: 4
        :param cpp: 48
        :param cp: 48
        :param c: 16
        :param reduction: indicates whether to reduce the output maps width
        :param reduction_prev: when previous cell reduced width, s1_d = s0_d//2
        in order to keep same shape between s1 and s0, we adopt prep0 layer to
        reduce the s0 width by half.
        """
        super(Cell, self).__init__()
        self.preprocess0 = ConvBlock(cpp, c, 1, 1, 0, affine=False)
        # preprocess1 deal with output from prev cell
        self.preprocess1 = ConvBlock(cp, c, 1, 1, 0, affine=False)

        # steps inside a cell
        self.num_nodes = num_nodes
        self.multiplier = multiplier

        self.layers = nn.ModuleList()

        for i in range(self.num_nodes):
            # for each i inside cell, it connects with all previous output
            # plus previous two cells' output
            for j in range(2 + i):
                # for reduction cell, it will reduce the heading 2 inputs only
                stride = 1
                layer = MixedLayer(c, stride, primitives, ops)
                self.layers.append(layer)

    def forward(self, s0, s1, weights):
        """
        :param s0:
        :param s1:
        :param weights: [14, 8]
        :return:
        """
        # print('s0:', s0.shape,end='=>')
        s0 = self.preprocess0(s0)  # [40, 48, 32, 32], [40, 16, 32, 32]
        # print(s0.shape, self.reduction_prev)
        # print('s1:', s1.shape,end='=>')
        s1 = self.preprocess1(s1)  # [40, 48, 32, 32], [40, 16, 32, 32]
        # print(s1.shape)

        states = [s0, s1]
        offset = 0
        # for each node, receive input from all previous intermediate nodes and s0, s1
        for i in range(self.num_nodes):  # 4
            # [40, 16, 32, 32]
            s = sum(self.layers[offset + j](h, weights[offset + j]) for j, h in enumerate(states))
            offset += len(states)
            # append one state since s is the elem-wise addition of all output
            states.append(s)
            # print('node:',i, s.shape, self.reduction)

        # concat along dim=channel
        return torch.cat(states[-self.multiplier:], dim=1)  # 6 of [40, 16, 32, 32]
