from __future__ import absolute_import

import torch
import torch.nn
import torch.nn.init
import torch.optim
import torch.nn.functional as F

from default_utils import set_seed as set_seed_defaultUtils


def set_parallelism_threads():  # for compatibility
    pass


def set_seed(seed):
    """ Set the random number seed to the desired value

        Parameters
        ----------
        seed : integer
            Random number seed.
    """

    set_seed_defaultUtils(seed)
    torch.manual_seed(seed)


def get_function(name):
    mapping = {}

    # loss
    mapping['mse'] = torch.nn.MSELoss()
    mapping['binary_crossentropy'] = torch.nn.BCELoss()
    mapping['categorical_crossentropy'] = torch.nn.CrossEntropyLoss()
    mapping['smoothL1'] = torch.nn.SmoothL1Loss()

    mapped = mapping.get(name)
    if not mapped:
        raise Exception('No pytorch function found for "{}"'.format(name))

    return mapped


def build_activation(activation):

    # activation
    if activation == 'relu':
        return torch.nn.ReLU()
    elif activation == 'sigmoid':
        return torch.nn.Sigmoid()
    elif activation == 'tanh':
        return torch.nn.Tanh()


def build_optimizer(model, optimizer, lr, kerasDefaults, trainable_only=True):
    if trainable_only:
        params = filter(lambda p: p.requires_grad, model.parameters())
    else:
        params = model.parameters()

    # schedule = optimizers.optimizer.Schedule() # constant lr (equivalent to default keras setting)

    if optimizer == 'sgd':
        return torch.optim.GradientDescentMomentum(params,
                                                   lr=lr,
                                                   momentum_coef=kerasDefaults['momentum_sgd'],
                                                   nesterov=kerasDefaults['nesterov_sgd'])

    elif optimizer == 'rmsprop':
        return torch.optim.RMSprop(model.parameters(),
                                   lr=lr,
                                   alpha=kerasDefaults['rho'],
                                   eps=kerasDefaults['epsilon'])

    elif optimizer == 'adagrad':
        return torch.optim.Adagrad(model.parameters(),
                                   lr=lr,
                                   eps=kerasDefaults['epsilon'])

    elif optimizer == 'adadelta':
        return torch.optim.Adadelta(params,
                                    eps=kerasDefaults['epsilon'],
                                    rho=kerasDefaults['rho'])

    elif optimizer == 'adam':
        return torch.optim.Adam(params,
                                lr=lr,
                                betas=[kerasDefaults['beta_1'], kerasDefaults['beta_2']],
                                eps=kerasDefaults['epsilon'])


def initialize(weights, initializer, kerasDefaults, seed=None, constant=0.):

    if initializer == 'constant':
        return torch.nn.init.constant_(weights, val=constant)

    elif initializer == 'uniform':
        return torch.nn.init.uniform(weights,
                                     a=kerasDefaults['minval_uniform'],
                                     b=kerasDefaults['maxval_uniform'])

    elif initializer == 'normal':
        return torch.nn.init.normal(weights,
                                    mean=kerasDefaults['mean_normal'],
                                    std=kerasDefaults['stddev_normal'])

    elif initializer == 'glorot_normal':  # not quite Xavier
        return torch.nn.init.xavier_normal(weights)

    elif initializer == 'glorot_uniform':
        return torch.nn.init.xavier_uniform_(weights)

    elif initializer == 'he_normal':
        return torch.nn.init.kaiming_uniform(weights)


def xent(y_true, y_pred):
    return F.cross_entropy(y_pred, y_true)


def mse(y_true, y_pred):
    return F.mse_loss(y_pred, y_true)
