from __future__ import print_function

import os
import sys
import logging

import pandas as pd
import numpy as np

file_path = os.path.dirname(os.path.realpath(__file__))
lib_path2 = os.path.abspath(os.path.join(file_path, '..', '..', 'common'))
sys.path.append(lib_path2)

import candle

logger = logging.getLogger(__name__)
candle.set_parallelism_threads()

additional_definitions = [
    {'name': 'latent_dim',
     'action': 'store',
     'type': int,
     'help': 'latent dimensions'},
    {'name': 'residual',
     'type': candle.str2bool,
     'default': False,
     'help': 'add skip connections to the layers'},
    {'name': 'reduce_lr',
     'type': candle.str2bool,
     'default': False,
     'help': 'reduce learning rate on plateau'},
    {'name': 'warmup_lr',
     'type': candle.str2bool,
     'default': False,
     'help': 'gradually increase learning rate on start'},
    {'name': 'base_lr',
     'type': float,
     'help': 'base learning rate'},
    {'name': 'epsilon_std',
     'type': float,
     'help': 'epsilon std for sampling latent noise'},
    {'name': 'use_cp',
     'type': candle.str2bool,
     'default': False,
     'help': 'checkpoint models with best val_loss'},
    {'name': 'use_tb',
     'type': candle.str2bool,
     'default': False,
     'help': 'use tensorboard'},
    {'name': 'tsne',
     'type': candle.str2bool,
     'default': False,
     'help': 'generate tsne plot of the latent representation'}
]

required = [
    'activation',
    'batch_size',
    'dense',
    'dropout',
    'epochs',
    'initialization',
    'learning_rate',
    'loss',
    'optimizer',
    'rng_seed',
    'scaling',
    'val_split',
    'latent_dim',
    'batch_normalization',
    'epsilon_std',
    'timeout'
]


class BenchmarkAttn(candle.Benchmark):

    def set_locals(self):
        """Functionality to set variables specific for the benchmark
        - required: set of required parameters for the benchmark.
        - additional_definitions: list of dictionaries describing the additional parameters for the
        benchmark.
        """

        if required is not None:
            self.required = set(required)
        if additional_definitions is not None:
            self.additional_definitions = additional_definitions


def extension_from_parameters(params, framework=''):
    """Construct string for saving model with annotation of parameters"""
    ext = framework
    for i, n in enumerate(params['dense']):
        if n:
            ext += '.D{}={}'.format(i + 1, n)
    ext += '.A={}'.format(params['activation'][0])
    ext += '.B={}'.format(params['batch_size'])
    ext += '.E={}'.format(params['epochs'])
    ext += '.L={}'.format(params['latent_dim'])
    ext += '.LR={}'.format(params['learning_rate'])
    ext += '.S={}'.format(params['scaling'])

    if params['epsilon_std'] != 1.0:
        ext += '.EPS={}'.format(params['epsilon_std'])
    if params['dropout']:
        ext += '.DR={}'.format(params['dropout'])
    if params['batch_normalization']:
        ext += '.BN'
    if params['warmup_lr']:
        ext += '.WU_LR'
    if params['reduce_lr']:
        ext += '.Re_LR'
    if params['residual']:
        ext += '.Res'

    return ext


def load_data(params, seed):

    # start change #
    if params['train_data'].endswith('h5') or params['train_data'].endswith('hdf5'):
        print('processing h5 in file {}'.format(params['train_data']))

        url = params['data_url']
        file_train = params['train_data']
        train_file = candle.get_file(file_train, url + file_train, cache_subdir='Pilot1')

        df_x_train_0 = pd.read_hdf(train_file, 'x_train_0').astype(np.float32)
        df_x_train_1 = pd.read_hdf(train_file, 'x_train_1').astype(np.float32)
        X_train = pd.concat([df_x_train_0, df_x_train_1], axis=1, sort=False)
        del df_x_train_0, df_x_train_1

        df_x_test_0 = pd.read_hdf(train_file, 'x_test_0').astype(np.float32)
        df_x_test_1 = pd.read_hdf(train_file, 'x_test_1').astype(np.float32)
        X_test = pd.concat([df_x_test_0, df_x_test_1], axis=1, sort=False)
        del df_x_test_0, df_x_test_1

        df_x_val_0 = pd.read_hdf(train_file, 'x_val_0').astype(np.float32)
        df_x_val_1 = pd.read_hdf(train_file, 'x_val_1').astype(np.float32)
        X_val = pd.concat([df_x_val_0, df_x_val_1], axis=1, sort=False)
        del df_x_val_0, df_x_val_1

        Y_train = pd.read_hdf(train_file, 'y_train')
        Y_test = pd.read_hdf(train_file, 'y_test')
        Y_val = pd.read_hdf(train_file, 'y_val')

        # assumes AUC is in the third column at index 2
        # df_y = df['AUC'].astype('int')
        # df_x = df.iloc[:,3:].astype(np.float32)

        # assumes dataframe has already been scaled
        # scaler = StandardScaler()
        # df_x = scaler.fit_transform(df_x)
    else:
        print('expecting in file file suffix h5')
        sys.exit()

    print('x_train shape:', X_train.shape)
    print('x_test shape:', X_test.shape)

    return X_train, Y_train, X_val, Y_val, X_test, Y_test

    # start change #
    if train_file.endswith('h5') or train_file.endswith('hdf5'):
        print('processing h5 in file {}'.format(train_file))

        df_x_train_0 = pd.read_hdf(train_file, 'x_train_0').astype(np.float32)
        df_x_train_1 = pd.read_hdf(train_file, 'x_train_1').astype(np.float32)
        X_train = pd.concat([df_x_train_0, df_x_train_1], axis=1, sort=False)
        del df_x_train_0, df_x_train_1

        df_x_test_0 = pd.read_hdf(train_file, 'x_test_0').astype(np.float32)
        df_x_test_1 = pd.read_hdf(train_file, 'x_test_1').astype(np.float32)
        X_test = pd.concat([df_x_test_0, df_x_test_1], axis=1, sort=False)
        del df_x_test_0, df_x_test_1

        df_x_val_0 = pd.read_hdf(train_file, 'x_val_0').astype(np.float32)
        df_x_val_1 = pd.read_hdf(train_file, 'x_val_1').astype(np.float32)
        X_val = pd.concat([df_x_val_0, df_x_val_1], axis=1, sort=False)
        del df_x_val_0, df_x_val_1

        Y_train = pd.read_hdf(train_file, 'y_train')
        Y_test = pd.read_hdf(train_file, 'y_test')
        Y_val = pd.read_hdf(train_file, 'y_val')

        # assumes AUC is in the third column at index 2
        # df_y = df['AUC'].astype('int')
        # df_x = df.iloc[:,3:].astype(np.float32)

        # assumes dataframe has already been scaled
        # scaler = StandardScaler()
        # df_x = scaler.fit_transform(df_x)
    else:
        print('expecting in file file suffix h5')
        sys.exit()

    print('x_train shape:', X_train.shape)
    print('x_test shape:', X_test.shape)

    return X_train, Y_train, X_val, Y_val, X_test, Y_test
