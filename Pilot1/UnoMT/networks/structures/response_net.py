"""
    File Name:          UnoPytorch/response_net.py
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               8/17/18
    Python Version:     3.6.6
    File Description:

"""
import torch
import torch.nn as nn
from networks.structures.residual_block import ResBlock
from networks.initialization.weight_init import basic_weight_init


class RespNet(nn.Module):

    def __init__(self,
                 gene_latent_dim: int,
                 drug_latent_dim: int,

                 gene_encoder: nn.Module,
                 drug_encoder: nn.Module,

                 resp_layer_dim: int,
                 resp_num_layers_per_block: int,
                 resp_num_blocks: int,

                 resp_num_layers: int,

                 resp_dropout: float,
                 resp_activation: str):

        super(RespNet, self).__init__()

        self.__gene_encoder = gene_encoder
        self.__drug_encoder = drug_encoder

        # Layer construction ##################################################
        # Network for response prediction
        self.__resp_net = nn.Sequential()

        # Reduce the dimension
        self.__resp_net.add_module(
            'dense_0',
            nn.Linear(gene_latent_dim + drug_latent_dim + 1, resp_layer_dim))

        self.__resp_net.add_module('activation_0', nn.ReLU())

        # Residual blocks
        for i in range(resp_num_blocks):
            self.__resp_net.add_module(
                'residual_block_%d' % i,
                ResBlock(layer_dim=resp_layer_dim,
                         num_layers=resp_num_layers_per_block,
                         dropout=resp_dropout))

        # Dense layers
        for i in range(1, resp_num_layers + 1):
            self.__resp_net.add_module('dense_%d' % i,
                                       nn.Linear(resp_layer_dim,
                                                 resp_layer_dim))

            if resp_dropout > 0.:
                self.__resp_net.add_module('dropout_%d' % i,
                                           nn.Dropout(resp_dropout))

            self.__resp_net.add_module('res_relu_%d' % i, nn.ReLU())

        # Last layer for regression
        self.__resp_net.add_module('dense_out', nn.Linear(resp_layer_dim, 1))

        if resp_activation.lower() == 'sigmoid':
            self.__resp_net.add_module('activation', nn.Sigmoid())
        elif resp_activation.lower() == 'tanh':
            self.__resp_net.add_module('activation', nn.Tanh())
        else:
            pass

        # Weight Initialization ###############################################
        self.__resp_net.apply(basic_weight_init)

    def forward(self, rnaseq, drug_feature, concentration):
        return self.__resp_net(torch.cat((self.__gene_encoder(rnaseq),
                                          self.__drug_encoder(drug_feature),
                                          concentration), dim=1))
