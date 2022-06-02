"""
    File Name:          UnoPytorch/classification_net.py
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               8/21/18
    Python Version:     3.6.6
    File Description:

"""
import torch
import torch.nn as nn
from networks.initialization.weight_init import basic_weight_init


class ClfNet(nn.Module):

    def __init__(self,
                 encoder: nn.Module,
                 input_dim: int,

                 condition_dim: int,
                 layer_dim: int,
                 num_layers: int,
                 num_classes: int):

        super(ClfNet, self).__init__()

        self.__encoder = encoder

        self.__clf_net = nn.Sequential()

        prev_dim = input_dim + condition_dim

        for i in range(num_layers):

            self.__clf_net.add_module('dense_%d' % i,
                                      nn.Linear(prev_dim, layer_dim))
            prev_dim = layer_dim
            self.__clf_net.add_module('relu_%d' % i, nn.ReLU())

        self.__clf_net.add_module('dense_%d' % num_layers,
                                  nn.Linear(prev_dim, num_classes))
        self.__clf_net.add_module('logsoftmax_%d' % num_layers,
                                  nn.LogSoftmax(dim=1))

        # Weight Initialization ###############################################
        self.__clf_net.apply(basic_weight_init)

    def forward(self, samples, conditions=None):
        if conditions is None:
            return self.__clf_net(self.__encoder(samples))
        else:
            return self.__clf_net(torch.cat((self.__encoder(samples),
                                             conditions), dim=1))
