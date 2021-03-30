from __future__ import absolute_import

import argparse
try:
    import configparser
except ImportError:
    import ConfigParser as configparser
    
import helper_utils as hutils

basic_parameters = [
    #'config_file',
    # neon parser
    #'verbose', 'logfile', 'save_path', 'model_name', 'data_type', 'dense', 'rng_seed', 'epochs', 'batch_size',
    # general behavior
    #'train_bool', 'eval_bool', 'timeout',
    
    {'name': 'verbose',
        'type': hutils.str2bool,
        'default': False,
        'help': 'increase output verbosity'},
]

cyclic_learning_parameters = [
    {'name': 'clr_flag',
        'type': hutils.str2bool,
        'default': argparse.SUPPRESS,
        'help': 'CLR flag (boolean)'},
    {'name': 'clr_mode',
        'type': str,
        'default': argparse.SUPPRESS,
        'choices': ['trng1', 'trng2', 'exp'],
        'help': 'CLR mode (default: trng1)'},
    {'name': 'clr_base_lr',
        'type': float,
        'default': argparse.SUPPRESS,
        'help': 'Base lr for cycle lr.'},
    {'name': 'clr_max_lr',
        'type': float,
        'default': argparse.SUPPRESS,
        'help': 'Max lr for cycle lr.'},
    {'name': 'clr_gamma',
        'type': float,
        'default': argparse.SUPPRESS,
        'help': 'Gamma parameter for learning cycle LR.'}
]
