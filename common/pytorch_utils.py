from __future__ import absolute_import

import torch
import torch.nn
import torch.nn.init
import torch.optim
import torch.nn.functional as F

from default_utils import set_seed as set_seed_defaultUtils

def set_parallelism_threads(): # for compatibility
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


def build_activation(type):

    # activation
    
    if type=='relu':
         return torch.nn.ReLU()
    elif type=='sigmoid':
         return torch.nn.Sigmoid()
    elif type=='tanh':
         return torch.nn.Tanh()


def build_optimizer(model, type, lr, kerasDefaults, trainable_only=True):
    
    if trainable_only:
        params = filter(lambda p: p.requires_grad, model.parameters())
    else:
        params = model.parameters()
    
    #schedule = optimizers.optimizer.Schedule() # constant lr (equivalent to default keras setting)

    if type == 'sgd':
        return torch.optim.GradientDescentMomentum(params,
                                                  lr=lr,
                                                  momentum_coef=kerasDefaults['momentum_sgd'],
                                                  nesterov=kerasDefaults['nesterov_sgd'])
                                                  #schedule=schedule)

    elif type == 'rmsprop':
        return torch.optim.RMSprop(model.parameters(),
                                  lr=lr,
                                  alpha=kerasDefaults['rho'],
                                  eps=kerasDefaults['epsilon'])
                                  #schedule=schedule)

    elif type == 'adagrad':
        return torch.optim.Adagrad(model.parameters(),
                              lr=lr,
                              eps=kerasDefaults['epsilon'])

    elif type == 'adadelta':
        return torch.optim.Adadelta(params,
                              eps=kerasDefaults['epsilon'],
                              rho=kerasDefaults['rho'])

    elif type == 'adam':
        return torch.optim.Adam(params,
                               lr=lr,
                               betas=[kerasDefaults['beta_1'], kerasDefaults['beta_2']],
                               eps=kerasDefaults['epsilon'])



def initialize(weights, type, kerasDefaults, seed=None, constant=0.):
    
    if type == 'constant':
        return torch.nn.init.constant_(weights,
                                    val=constant)
    
    elif type == 'uniform':
        return torch.nn.init.uniform(weights,
                                  a=kerasDefaults['minval_uniform'],
                                  b=kerasDefaults['maxval_uniform'])

    elif type == 'normal':
        return torch.nn.init.normal(weights,
                                  mean=kerasDefaults['mean_normal'],
                                  std=kerasDefaults['stddev_normal'])

    elif type == 'glorot_normal': # not quite Xavier
        return torch.nn.init.xavier_normal(weights)

    elif type == 'glorot_uniform':
        return torch.nn.init.xavier_uniform_(weights)

    elif type == 'he_normal':
        return torch.nn.init.kaiming_uniform(weights)


def xent(y_true, y_pred):
    return F.cross_entropy(y_pred, y_true)


def mse(y_true, y_pred):
    return F.mse_loss(y_pred, y_true)
