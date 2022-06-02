"""
CNN NLP operations closely modeled after the original paper's vision task.
"""

import torch
import torch.nn as nn

from darts.api import Model


OPS = {
    'none': lambda c, stride, affine: Zero(stride),
    'avg_pool_3': lambda c, stride, affine: nn.AvgPool1d(3, stride=stride, padding=1, count_include_pad=False),
    'max_pool_3': lambda c, stride, affine: nn.MaxPool1d(3, stride=stride, padding=1),
    'skip_connect': lambda c, stride, affine: Identity() if stride == 1 else FactorizedReduce(c, c, affine=affine),
    'sep_conv_3': lambda c, stride, affine: SepConv(c, c, 3, stride, 1, affine=affine),
    'sep_conv_5': lambda c, stride, affine: SepConv(c, c, 5, stride, 2, affine=affine),
    'sep_conv_7': lambda c, stride, affine: SepConv(c, c, 7, stride, 3, affine=affine),
    'dil_conv_3': lambda c, stride, affine: DilConv(c, c, 3, stride, 2, 2, affine=affine),
    'dil_conv_5': lambda c, stride, affine: DilConv(c, c, 5, stride, 4, 2, affine=affine),
    'convblock_7': lambda c, stride, affine: ConvBlock(c, c, 7, stride, 3, affine=affine),
}


class ConvBlock(Model):
    """ ReLu -> Conv1d -> BatchNorm """

    def __init__(self, c_in, c_out, kernel_size, stride, padding, affine=True):
        super(ConvBlock, self).__init__()

        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv1d(c_in, c_out, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm1d(c_out, affine=affine)
        )

    def forward(self, x):
        return self.op(x)


class DilConv(Model):
    """ ReLU Dilated Convolution """

    def __init__(self, c_in, c_out, kernel_size, stride, padding, dilation, affine=True):
        super(DilConv, self).__init__()

        self.op = nn.Sequential(
            nn.ReLU(inplace=False),

            nn.Conv1d(
                c_in,
                c_in,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=c_in,
                bias=False
            ),

            nn.Conv1d(
                c_in,
                c_out,
                kernel_size=1,
                padding=0,
                bias=False
            ),

            nn.BatchNorm1d(c_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x)


class FactorizedReduce(Model):
    """ Reduce the feature maps by half, maintaining number of channels

    Example
    -------
    x: torch.Size([2, 10, 12])
    out: [batch_size, c_out, d//2]
    out: torch.Size([2, 10, 6])
    """

    def __init__(self, c_in, c_out, affine=True):
        super(FactorizedReduce, self).__init__()
        assert c_out % 2 == 0

        self.conv_1 = nn.Conv1d(c_in, c_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv_2 = nn.Conv1d(c_in, c_out // 2, 1, stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm1d(c_out, affine=affine)

    def forward(self, x):
        x = torch.relu(x)
        out = torch.cat([self.conv_1(x), self.conv_2(x[:, :, 1:])], dim=1)
        out = self.bn(out)
        return out


class Identity(Model):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class SepConv(Model):
    """ Separable Convolution Block """
    def __init__(self, c_in, c_out, kernel_size, stride, padding, affine=True):
        super(SepConv, self).__init__()

        self.op = nn.Sequential(
            nn.ReLU(inplace=False),

            nn.Conv1d(
                c_in,
                c_in,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=c_in,
                bias=False
            ),

            nn.Conv1d(
                c_in,
                c_in,
                kernel_size=1,
                padding=0,
                bias=False
            ),

            nn.BatchNorm1d(c_in, affine=affine),
            nn.ReLU(inplace=False),

            nn.Conv1d(
                c_in,
                c_in,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                groups=c_in,
                bias=False
            ),

            nn.Conv1d(c_in, c_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm1d(c_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x)


class Zero(nn.Module):
    """ Zero tensor by stride """

    def __init__(self, stride):
        super(Zero, self).__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return x.mul(0.)
        return x[:, :, ::self.stride].mul(0.)
