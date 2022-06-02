"""
    File Name:          UnoPytorch/optimizer.py
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               8/22/18
    Python Version:     3.6.6
    File Description:

"""
import collections

import torch.nn as nn
from torch.optim import Adam, RMSprop, SGD


def get_optimizer(
        opt_type: str,
        networks: nn.Module or iter,
        learning_rate: float,
        l2_regularization: float):

    if isinstance(networks, collections.Iterable):
        params = []
        for n in networks:
            params += list(n.parameters())
        params = list(set(params))

    else:
        params = networks.parameters()

    if opt_type.lower() == 'adam':
        optimizer = Adam(params,
                         lr=learning_rate,
                         amsgrad=True,
                         weight_decay=l2_regularization)
    elif opt_type.lower() == 'rmsprop':
        optimizer = RMSprop(params,
                            lr=learning_rate,
                            weight_decay=l2_regularization)
    else:
        optimizer = SGD(params,
                        lr=learning_rate,
                        momentum=0.9,
                        weight_decay=l2_regularization)

    return optimizer
