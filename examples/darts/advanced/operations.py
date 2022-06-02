import torch.nn as nn
import torch.nn.functional as F


""" DARTS operations contstructor """
OPS = {
    'none': lambda c, stride, affine: Identity(),
    'conv_3': lambda c, stride, affine: ConvBlock(c, c, 3, stride),
    'dil_conv': lambda c, stride, affine: DilConv(c, c, 3, stride, 2, 2, affine=affine)
}


class Stem(nn.Module):
    """ Network stem

    This will always be the beginning of the network.
    DARTS will only recompose modules after the stem.
    For this reason, we define this separate from the
    other modules in the network.

    Args:
        input_dim: the input dimension for your data

        cell_dim: the intermediate dimension size for
                  the remaining modules of the network.
    """
    def __init__(self, in_channels: int = 1, cell_dim: int = 100, kernel_size=3):
        super(Stem, self).__init__()
        self.stem = nn.Conv2d(in_channels, cell_dim, kernel_size)

    def forward(self, x):
        x = self.stem(x)
#        print(f'stem: {x.shape}')
        return x


class ConvBlock(nn.Module):
    """ ReLu -> Conv2d """

    def __init__(self, c_in, c_out, kernel_size, stride, affine=True):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(
            c_in, c_out, kernel_size=kernel_size, stride=stride
        )

    def forward(self, x):
        return self.conv(F.relu(x))


class DilConv(nn.Module):
    """ ReLU Dilated Convolution """

    def __init__(self, c_in, c_out, kernel_size,
                 stride, padding, dilation, affine=True):
        super(DilConv, self).__init__()

        self.op = nn.Sequential(
            nn.ReLU(inplace=False),

            nn.Conv2d(
                c_in,
                c_in,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=c_in,
                bias=False
            ),

            nn.Conv2d(
                c_in,
                c_out,
                kernel_size=1,
                padding=0,
                bias=False
            ),

            nn.BatchNorm2d(c_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x)


class Identity(nn.Module):
    """ Identity module """

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x
