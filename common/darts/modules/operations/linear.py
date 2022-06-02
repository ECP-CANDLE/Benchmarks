"""
Linear operations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from darts.api import Model


OPS = {
    'none': lambda c, stride, affice: Zero(),
    'skip_connect': lambda c, stride, affine: Identity(),
    'linear_block': lambda c, stride, affine: LinearBlock(c, c, affine=affine),
    'linear_conv': lambda c, stride, affine: LinearConv(c, c, 1),
    'linear_drop': lambda c, stride, affine: LinearDrop(c, c, 1),
    'encoder': lambda c, stride, affine: Encoder(c, c, 1),
}


class LinearBlock(Model):
    """ Linear block consisting of two fully connected layers

    Example
    -------
    x: torch.Size([2, 10, 12])
    out: [batch_size, c_out, d//2]
    out: torch.Size([2, 10, 6])
    """

    def __init__(self, c_in, c_out, affine=True):
        super(LinearBlock, self).__init__()
        assert c_out % 2 == 0

        self.fc1 = nn.Linear(c_in, c_in * 2)
        self.fc2 = nn.Linear(c_in * 2, c_out)

    def forward(self, x):
        x = torch.relu(x)
        x = self.fc1(x)
        out = self.fc2(x)
        return out


class LinearDrop(Model):
    """ Linear block with dropout """

    def __init__(self, c_in, c_out, affine=True):
        super(LinearDrop, self).__init__()
        assert c_out % 2 == 0

        self.fc1 = nn.Linear(c_in, c_in * 2)
        self.fc2 = nn.Linear(c_in * 2, c_out)

    def forward(self, x):
        x = torch.relu(x)
        x = F.dropout(self.fc1(x))
        out = F.dropout(self.fc2(x))
        return out


class Encoder(Model):
    """ Linear encoder """

    def __init__(self, c_in, c_out, affine=True):
        super(Encoder, self).__init__()
        assert c_out % 2 == 0

        self.fc1 = nn.Linear(c_in, c_in // 2)
        self.fc2 = nn.Linear(c_in // 2, c_in)

    def forward(self, x):
        x = torch.relu(x)
        x = self.fc1(x)
        return self.fc2(x)


class LinearConv(Model):
    """ Linear => Conv => Linear """

    def __init__(self, c_in, c_out, kernel_size):
        super(LinearConv, self).__init__()
        self.fc_1 = nn.Linear(c_in, c_in)
        self.conv = nn.Conv1d(c_in, c_in, kernel_size)
        self.fc_2 = nn.Linear(c_in, c_out)

    def forward(self, x):
        x = torch.relu(x)
        x = self.fc_1(x)
        x = self.conv(x)
        return x


class Identity(Model):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Zero(nn.Module):
    """ Zero tensor by stride """

    def __init__(self):
        super(Zero, self).__init__()

    def forward(self, x):
        return x
