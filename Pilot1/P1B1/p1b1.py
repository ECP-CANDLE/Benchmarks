from __future__ import print_function

import os
import sys
import logging

import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from scipy.stats.stats import pearsonr

file_path = os.path.dirname(os.path.realpath(__file__))
#lib_path = os.path.abspath(os.path.join(file_path, '..'))
#sys.path.append(lib_path)
lib_path2 = os.path.abspath(os.path.join(file_path, '..', '..', 'common'))
sys.path.append(lib_path2)

import candle

logger = logging.getLogger(__name__)
candle.set_parallelism_threads()

additional_definitions = [ 
{'name':'latent_dim', 
    'action':'store',
    'type': int, 
    'help':'latent dimensions'},
{'name':'model', 
    'default':'ae',
    'choices':['ae', 'vae', 'cvae'],
    'help':'model to use: ae, vae, cvae'},
{'name':'use_landmark_genes', 
    'type': candle.str2bool,
    'default': False,
    'help':'use the 978 landmark genes from LINCS (L1000) as expression features'},
{'name':'residual', 
    'type': candle.str2bool,
    'default': False,
    'help':'add skip connections to the layers'},
{'name':'reduce_lr', 
    'type': candle.str2bool,
    'default': False,
    'help':'reduce learning rate on plateau'},
{'name':'warmup_lr', 
    'type': candle.str2bool,
    'default': False,
    'help':'gradually increase learning rate on start'},
{'name':'base_lr', 
    'type': float,
    'help':'base learning rate'},
{'name':'epsilon_std', 
    'type': float,
    'help':'epsilon std for sampling latent noise'},
{'name':'cp', 
    'type': candle.str2bool,
    'default': False, 
    'help':'checkpoint models with best val_loss'},
#{'name':'shuffle', 
    #'type': candle.str2bool,
    #'default': False, 
    #'help':'shuffle data'},
{'name':'tb', 
    'type': candle.str2bool,
    'default': False, 
    'help':'use tensorboard'},
{'name':'tsne', 
    'type': candle.str2bool,
    'default': False, 
    'help':'generate tsne plot of the latent representation'}
]

required = [
    'activation',
    'batch_size',
    'dense',
    'drop',
    'epochs',
    'initialization',
    'learning_rate',
    'loss',
    'noise_factor',
    'optimizer',
    'rng_seed',
    'model',
    'scaling',
    'validation_split',
    'latent_dim',
    'feature_subsample',
    'batch_normalization',
    'epsilon_std',
    'solr_root',
    'timeout'
    ]

class BenchmarkP1B1(candle.Benchmark):

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
    ext += '.{}'.format(params['model'])
    for i, n in enumerate(params['dense']):
        if n:
            ext += '.D{}={}'.format(i+1, n)
    ext += '.A={}'.format(params['activation'])
    ext += '.B={}'.format(params['batch_size'])
    ext += '.E={}'.format(params['epochs'])
    ext += '.L={}'.format(params['latent_dim'])
    ext += '.LR={}'.format(params['learning_rate'])
    ext += '.S={}'.format(params['scaling'])
    if params['epsilon_std'] != 1.0:
        ext += '.EPS={}'.format(params['epsilon_std'])
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
    if params['warmup_lr']:
        ext += '.WU_LR'
    if params['reduce_lr']:
        ext += '.Re_LR'
    if params['residual']:
        ext += '.Res'

    return ext


def load_data(params, seed):
    drop_cols = ['case_id']
    onehot_cols = ['cancer_type']
    y_cols = ['cancer_type']

    if params['use_landmark_genes']:
        lincs_file = 'lincs1000.tsv'
        lincs_path = candle.fetch_file(params['url_p1b1'] + lincs_file, 'Pilot1')
        df_l1000 = pd.read_csv(lincs_path, sep='\t')
        x_cols = df_l1000['gdc'].tolist()
        drop_cols = None
    else:
        x_cols = None

    train_path = candle.fetch_file(params['url_p1b1'] + params['file_train'], 'Pilot1')
    test_path = candle.fetch_file(params['url_p1b1'] + params['file_test'], 'Pilot1')

    return candle.load_csv_data(train_path, test_path,
                                   x_cols=x_cols,
                                   y_cols=y_cols,
                                   drop_cols=drop_cols,
                                   onehot_cols=onehot_cols,
                                   n_cols=params['feature_subsample'],
                                   shuffle=params['shuffle'],
                                   scaling=params['scaling'],
                                   dtype=params['datatype'],
                                   validation_split=params['validation_split'],
                                   return_dataframe=False,
                                   return_header=True,
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
        lincs_path = candle.fetch_file(url_p1b1 + lincs_file)
        df_l1000 = pd.read_csv(lincs_path, sep='\t')
        usecols = df_l1000['gdc']
        drop_cols = None
    else:
        usecols = None

    return candle.load_X_data(params['url_p1b1'], params['file_train'], params['file_test'],
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
    try:
        mse = mean_squared_error(y_pred, y_test)
        r2 = r2_score(y_test, y_pred)
        corr, _ = pearsonr(y_pred.flatten(), y_test.flatten())
        # print('Mean squared error: {}%'.format(mse))
    except:
        #when nan or something else breaks mean_squared_error computation
        # we may check earlier before computation also:
        #np.isnan(y_pred).any() or np.isnan(y_test).any()):
        r2 = 0
        mse = 0
        corr = 0
    return {'mse': mse, 'r2_score': r2, 'correlation': corr}
