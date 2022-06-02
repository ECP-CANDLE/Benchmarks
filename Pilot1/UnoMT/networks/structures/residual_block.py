"""
    File Name:          UnoPytorch/residual_block.py
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               8/13/18
    Python Version:     3.6.6
    File Description:

"""
import torch.nn as nn
from networks.initialization.weight_init import basic_weight_init


class ResBlock(nn.Module):

    def __init__(self,
                 layer_dim: int,
                 num_layers: int,
                 dropout: float):

        super(ResBlock, self).__init__()

        # Layer construction ##################################################
        self.block = nn.Sequential()

        for i in range(num_layers):

            self.block.add_module('res_dense_%d' % i,
                                  nn.Linear(layer_dim, layer_dim))

            if dropout > 0.:
                self.block.add_module('res_dropout_%d' % i,
                                      nn.Dropout(dropout))

            if i != (num_layers - 1):
                self.block.add_module('res_relu_%d' % i, nn.ReLU())

        self.activation = nn.ReLU()

        # Weight Initialization ###############################################
        self.block.apply(basic_weight_init)

    def forward(self, x):
        return self.activation(self.block(x) + x)


if __name__ == '__main__':

    res_block = ResBlock(
        layer_dim=200,
        num_layers=2,
        dropout=0.2)

    print(res_block)
