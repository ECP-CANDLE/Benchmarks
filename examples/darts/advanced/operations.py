import torch.nn as nn
import torch.nn.functional as F

""" DARTS operations contstructor """
OPS = {
    'mlp'  : lambda c, stride, affine: MLP(c, c),
    'conv' : lambda c, stride, affine: ConvBlock(c, c, 3, stride, 1, affine=affine),
}


class StemNet(nn.Module):
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
    def __init__(self, input_dim: int=250, cell_dim: int=100):
        super(StemNet, self).__init__()
        self.fc = nn.Linear(input_dim, cell_dim)

    def forward(self, x):
        return self.fc(x)


class MLP(nn.Module):
    """ Multi-layer perceptron """

    def __init__(self, cell_dim, hidden_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(cell_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, cell_dim)

    def forward(self, x):
        return self.fc2(self.fc1(F.relu(x)))


class ConvBlock(nn.Module):
    """ ReLu -> Conv1d -> BatchNorm """

    def __init__(self, c_in, c_out, kernel_size,
                 stride, padding, affine=True):
        super(ConvBlock, self).__init__()

        self.op = nn.Sequential(
            #nn.ReLU(inplace=False),
            nn.Conv2d(
                c_in, c_out, kernel_size,
                stride=stride, padding=padding, bias=False),
            nn.BatchNorm1d(c_out, affine=affine)
        )

    def forward(self, x):
        return self.op(x)
