import torch
import torch.nn as nn

from darts.api import Model
from darts.modules.mixed_layer import MixedLayer
from darts.modules.operations.original import ConvBlock, FactorizedReduce


class Cell(Model):

    def __init__(self, num_nodes, multiplier, cpp, cp, c, reduction, reduction_prev):
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

        # indicating current cell is reduction or not
        self.reduction = reduction
        self.reduction_prev = reduction_prev

        # preprocess0 deal with output from prev_prev cell
        if reduction_prev:
            # if prev cell has reduced channel/double width,
            # it will reduce width by half
            self.preprocess0 = FactorizedReduce(cpp, c, affine=False)
        else:
            self.preprocess0 = ConvBlock(cpp, c, 1, 1, 0, affine=False)
        # preprocess1 deal with output from prev cell
        self.preprocess1 = ConvBlock(cp, c, 1, 1, 0, affine=False)

        # steps inside a cell
        self.num_nodes = num_nodes # 4
        self.multiplier = multiplier # 4

        self.layers = nn.ModuleList()

        for i in range(self.num_nodes):
            # for each i inside cell, it connects with all previous output
            # plus previous two cells' output
            for j in range(2 + i):
                # for reduction cell, it will reduce the heading 2 inputs only
                stride = 2 if reduction and j < 2 else 1
                layer = MixedLayer(c, stride)
                self.layers.append(layer)

    def forward(self, s0, s1, weights):
        """
        :param s0:
        :param s1:
        :param weights: [14, 8]
        :return:
        """
        #print('s0:', s0.shape,end='=>')
        s0 = self.preprocess0(s0) # [40, 48, 32, 32], [40, 16, 32, 32]
        #print(s0.shape, self.reduction_prev)
        #print('s1:', s1.shape,end='=>')
        s1 = self.preprocess1(s1) # [40, 48, 32, 32], [40, 16, 32, 32]
        #print(s1.shape)

        states = [s0, s1]
        offset = 0
        # for each node, receive input from all previous intermediate nodes and s0, s1
        for i in range(self.num_nodes): # 4
            # [40, 16, 32, 32]
            s = sum(self.layers[offset + j](h, weights[offset + j]) for j, h in enumerate(states))
            offset += len(states)
            # append one state since s is the elem-wise addition of all output
            states.append(s)
            #print('node:',i, s.shape, self.reduction)

        # concat along dim=channel
        return torch.cat(states[-self.multiplier:], dim=1) # 6 of [40, 16, 32, 32]