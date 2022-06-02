"""
    File Name:          UnoPytorch/encoder_net.py
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               8/13/18
    Python Version:     3.6.6
    File Description:

"""
import torch.nn as nn
from networks.initialization.weight_init import basic_weight_init


class EncNet(nn.Module):

    def __init__(self,
                 input_dim: int,
                 layer_dim: int,
                 num_layers: int,
                 latent_dim: int,
                 autoencoder: bool = True):

        super(EncNet, self).__init__()

        # Encoder #############################################################
        self.encoder = nn.Sequential()

        prev_dim = input_dim

        for i in range(num_layers):

            self.encoder.add_module('dense_%d' % i,
                                    nn.Linear(prev_dim, layer_dim))
            prev_dim = layer_dim
            self.encoder.add_module('relu_%d' % i, nn.ReLU())

        self.encoder.add_module('dense_%d' % num_layers,
                                nn.Linear(prev_dim, latent_dim))

        # self.encoder.add_module('activation', nn.Tanh())

        # Decoder #############################################################
        if autoencoder:

            self.decoder = nn.Sequential()

            prev_dim = latent_dim

            for i in range(num_layers):
                self.decoder.add_module('dense_%d' % i,
                                        nn.Linear(prev_dim, layer_dim))
                prev_dim = layer_dim
                self.decoder.add_module('relu_%d' % i, nn.ReLU())

            self.decoder.add_module('dense_%d' % num_layers,
                                    nn.Linear(prev_dim, input_dim))

        else:
            self.decoder = None

        # Weight Initialization ###############################################
        self.encoder.apply(basic_weight_init)
        if self.decoder is not None:
            self.decoder.apply(basic_weight_init)

    def forward(self, samples):

        if self.decoder is None:
            return self.encoder(samples)
        else:
            return self.decoder(self.encoder(samples))


if __name__ == '__main__':

    ent = EncNet(
        input_dim=100,
        layer_dim=200,
        latent_dim=20,
        num_layers=2,
        autoencoder=True,)

    print(ent)
