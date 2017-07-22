from __future__ import print_function

import os
import sys
import logging
import argparse
try:
    import configparser
except ImportError:
    import ConfigParser as configparser

import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from scipy.stats.stats import pearsonr

file_path = os.path.dirname(os.path.realpath(__file__))
lib_path = os.path.abspath(os.path.join(file_path, '..', 'common'))
sys.path.append(lib_path)
lib_path2 = os.path.abspath(os.path.join(file_path, '..', '..', 'common'))
sys.path.append(lib_path2)

import p1_common

url_p1b1 = 'http://ftp.mcs.anl.gov/pub/candle/public/benchmarks/P1B1/'
file_train = 'P1B1.dev.train.csv'
file_test = 'P1B1.dev.test.csv'

logger = logging.getLogger(__name__)

def common_parser(parser):

    parser.add_argument("--config-file", dest='config_file', type=str,
                        default=os.path.join(file_path, 'p1b1_default_model.txt'),
                        help="specify model configuration file")

    # Parse has been split between arguments that are common with the default neon parser
    # and all the other options
    parser = p1_common.get_default_neon_parse(parser)
    parser = p1_common.get_p1_common_parser(parser)

    # Arguments that are applicable just to p1b1
    parser = p1b1_parser(parser)

    return parser


def p1b1_parser(parser):
    parser.add_argument("--latent_dim", type=int,
                        default=argparse.SUPPRESS,
                        help="latent dimensions")
    parser.add_argument("--vae", action='store_true',
                        help="variational autoencoder")
    parser.add_argument("--with_type", action='store_true',
                        help="include one-hot encoded type information")
    parser.add_argument("--use_landmark_genes", action="store_true",
                        help="use the 978 landmark genes from LINCS (L1000) as expression features")
    parser.add_argument("--residual", action="store_true",
                        help="add skip connections to the layers")

    return parser


def read_config_file(file):
    config = configparser.ConfigParser()
    config.read(file)
    section = config.sections()
    fileParams = {}
    fileParams['activation'] = eval(config.get(section[0],'activation'))
    fileParams['batch_size'] = eval(config.get(section[0],'batch_size'))
    fileParams['dense'] = eval(config.get(section[0],'dense'))
    fileParams['drop']=eval(config.get(section[0],'drop'))
    fileParams['epochs'] = eval(config.get(section[0],'epochs'))
    fileParams['initialization'] = eval(config.get(section[0],'initialization'))
    fileParams['learning_rate'] = eval(config.get(section[0], 'learning_rate'))
    fileParams['loss'] = eval(config.get(section[0],'loss'))
    fileParams['noise_factor'] = eval(config.get(section[0],'noise_factor'))
    fileParams['optimizer'] = eval(config.get(section[0],'optimizer'))
    fileParams['rng_seed'] = eval(config.get(section[0],'rng_seed'))
    fileParams['scaling'] = eval(config.get(section[0],'scaling'))
    fileParams['validation_split'] = eval(config.get(section[0],'validation_split'))
    fileParams['latent_dim'] = eval(config.get(section[0],'latent_dim'))
    fileParams['feature_subsample'] = eval(config.get(section[0],'feature_subsample'))
    fileParams['batch_normalization'] = eval(config.get(section[0],'batch_normalization'))

    fileParams['solr_root'] = eval(config.get(section[1],'solr_root'))
    return fileParams


def extension_from_parameters(params, framework=''):
    """Construct string for saving model with annotation of parameters"""
    ext = framework
    for i, n in enumerate(params['dense']):
        if n:
            ext += '.D{}={}'.format(i+1, n)
    ext += '.A={}'.format(params['activation'])
    ext += '.B={}'.format(params['batch_size'])
    ext += '.E={}'.format(params['epochs'])
    ext += '.L={}'.format(params['latent_dim'])
    ext += '.R={}'.format(params['learning_rate'])
    ext += '.S={}'.format(params['scaling'])
    if params['feature_subsample'] > 0:
        ext += '.FS={}'.format(params['feature_subsample'])
    if params['drop']:
        ext += '.DR={}'.format(params['drop'])
    if params['alpha_dropout']:
        ext += '.AD'
    if params['batch_normalization']:
        ext += '.BN'
    if params['use_landmark_genes']:
        ext += '.L1000'
    if params['residual']:
        ext += '.Res'
    if params['vae']:
        ext += '.VAE'
    if params['with_type']:
        ext += '.WT'

    return ext


def load_data(params, seed):
    if params['with_type']:
        drop_cols = ['case_id']
        onehot_cols = ['cancer_type']
        y_cols = None
    else:
        drop_cols = ['case_id']
        onehot_cols = ['cancer_type']
        # onehot_cols = None
        y_cols = ['cancer_type']

    if params['use_landmark_genes']:
        lincs_file = 'lincs1000.tsv'
        lincs_path = p1_common.get_p1_file(url_p1b1 + lincs_file)
        df_l1000 = pd.read_csv(lincs_path, sep='\t')
        x_cols = df_l1000['gdc'].tolist()
        drop_cols = None
    else:
        x_cols = None

    train_path = p1_common.get_p1_file(url_p1b1 + file_train)
    test_path = p1_common.get_p1_file(url_p1b1 + file_test)

    return p1_common.load_csv_data(train_path, test_path,
                                   return_dataframe=False,
                                   y_cols=y_cols,
                                   x_cols=x_cols,
                                   drop_cols=drop_cols,
                                   onehot_cols=onehot_cols,
                                   n_cols=params['feature_subsample'],
                                   shuffle=params['shuffle'],
                                   scaling=params['scaling'],
                                   dtype=params['datatype'],
                                   validation_split=params['validation_split'],
                                   seed=seed)


def load_data_orig(params, seed):
    if params['with_type']:
        drop_cols = ['case_id']
        onehot_cols = ['cancer_type']
    else:
        drop_cols = ['case_id', 'cancer_type']
        onehot_cols = None

    if params['use_landmark_genes']:
        lincs_file = 'lincs1000.tsv'
        lincs_path = p1_common.get_p1_file(url_p1b1 + lincs_file)
        df_l1000 = pd.read_csv(lincs_path, sep='\t')
        usecols = df_l1000['gdc']
        drop_cols = None
    else:
        usecols = None

    return p1_common.load_X_data(url_p1b1, file_train, file_test,
                                 drop_cols=drop_cols,
                                 onehot_cols=onehot_cols,
                                 usecols=usecols,
                                 n_cols=params['feature_subsample'],
                                 shuffle=params['shuffle'],
                                 scaling=params['scaling'],
                                 validation_split=params['validation_split'],
                                 dtype=params['datatype'],
                                 seed=seed)


def evaluate_autoencoder(y_pred, y_test):
    mse = mean_squared_error(y_pred, y_test)
    r2 = r2_score(y_test, y_pred)
    corr, _ = pearsonr(y_pred.flatten(), y_test.flatten())
    # print('Mean squared error: {}%'.format(mse))
    return {'mse': mse, 'r2_score': r2, 'correlation': corr}
