from __future__ import print_function

import os
import sys
import logging
import argparse
try:
    import configparser
except ImportError:
    import ConfigParser as configparser

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats.stats import pearsonr

#file_path = os.path.dirname(os.path.realpath(__file__))
file_path = os.path.dirname(os.path.realpath(os.path.join(__file__, '..')))
lib_path = os.path.abspath(os.path.join(file_path, '..'))
sys.path.append(lib_path)
lib_path = os.path.abspath(os.path.join(file_path, 'data_utils_'))
sys.path.append(lib_path)
lib_path = os.path.abspath(os.path.join(file_path, 'model_utils_'))
sys.path.append(lib_path)
lib_path2 = os.path.abspath(os.path.join(file_path, '..', '..', 'common'))
sys.path.append(lib_path2)


import candle

P1B3_URL = 'http://ftp.mcs.anl.gov/pub/candle/public/benchmarks/P1B3/'
DATA_URL = 'http://ftp.mcs.anl.gov/pub/candle/public/benchmarks/Pilot1/combo/'

loggerUno = logging.getLogger(__name__)


def set_up_logger(logfile, logger1, logger2, verbose):
    candle.verify_path(logfile)
    fh = logging.FileHandler(logfile)
    fh.setFormatter(logging.Formatter("[%(asctime)s %(process)d] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    fh.setLevel(logging.DEBUG)

    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter(''))
    sh.setLevel(logging.DEBUG if verbose else logging.INFO)

    for log in [logger1, logger2]:
        log.setLevel(logging.DEBUG)
        log.addHandler(fh)
        log.addHandler(sh)


def extension_from_parameters(args):
    """Construct string for saving model with annotation of parameters"""
    ext = ''
    ext += '.A={}'.format(args.activation)
    ext += '.B={}'.format(args.batch_size)
    ext += '.E={}'.format(args.epochs)
    ext += '.O={}'.format(args.optimizer)
    ext += '.LS={}'.format(args.loss)
    # ext += '.LEN={}'.format(args.maxlen)
    ext += '.LR={}'.format(args.learning_rate)
    ext += '.CF={}'.format(''.join([x[0] for x in sorted(args.cell_features)]))
    ext += '.DF={}'.format(''.join([x[0] for x in sorted(args.drug_features)]))
    if args.feature_subsample > 0:
        ext += '.FS={}'.format(args.feature_subsample)
    if args.drop > 0:
        ext += '.DR={}'.format(args.drop)
    if args.warmup_lr:
        ext += '.wu_lr'
    if args.reduce_lr:
        ext += '.re_lr'
    if args.residual:
        ext += '.res'
    if args.use_landmark_genes:
        ext += '.L1000'
    if args.no_gen:
        ext += '.ng'
    for i, n in enumerate(args.dense):
        if n > 0:
            ext += '.D{}={}'.format(i+1, n)
    if args.dense_feature_layers != args.dense:
        for i, n in enumerate(args.dense):
            if n > 0:
                ext += '.FD{}={}'.format(i+1, n)

    return ext

def set_up_logger_data(verbose=False):
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter(''))
    sh.setLevel(logging.DEBUG if verbose else logging.INFO)

    logger.setLevel(logging.DEBUG)
    logger.addHandler(sh)


def log_evaluation(metric_outputs, logger, description='Comparing y_true and y_pred:'):
    logger.info(description)
    for metric, value in metric_outputs.items():
        logger.info('  {}: {:.4f}'.format(metric, value))


def get_file_p1(url):
    fname = os.path.basename(url)
    return candle.get_file(fname, origin=url, cache_subdir='Pilot1')


def dict_compare(d1, d2, ignore=[], expand=False):
    d1_keys = set(d1.keys()) - set(ignore)
    d2_keys = set(d2.keys()) - set(ignore)
    intersect_keys = d1_keys.intersection(d2_keys)
    added = d1_keys - d2_keys
    removed = d2_keys - d1_keys
    modified = set({x : (d1[x], d2[x]) for x in intersect_keys if d1[x] != d2[x]})
    common = set(x for x in intersect_keys if d1[x] == d2[x])
    equal = not (added or removed or modified)
    if expand:
        return equal, added, removed, modified, common
    else:
        return equal, added | removed | modified


def evaluate_prediction(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    corr, _ = pearsonr(y_true, y_pred)
    return {'mse': mse, 'mae': mae, 'r2': r2, 'corr': corr}


def read_IDs_file(fname):

    with open(fname, 'r') as f:
        read_ids = f.read().splitlines()
    
    loggerUno.info('Read file: {}'.format(fname))
    loggerUno.info('Number of elements read: {}'.format(len(read_ids)))

    return read_ids


class BenchmarkUno(candle.Benchmark):

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


additional_definitions = [
# Feature selection
    {'name':'agg_dose',
        'type': str,
        'default': None,
        'choices':['AUC', 'IC50', 'EC50', 'HS', 'AAC1', 'AUC1', 'DSS1'],
        'help':'use dose-independent response data with the specified aggregation metric'},
    {'name':'cell_features',
        'nargs':'+',
        'choices':['rnaseq', 'none'],
        'help':'use rnaseq cell line feature set or none at all'},
    {'name':'drug_features',
        'nargs':'+',
        'choices':['descriptors', 'fingerprints', 'none'],
        'help':'use dragon7 descriptors or fingerprint descriptors for drug features or none at all'},
    {'name': 'by_cell',
        'type':str,
        'default':None,
        'help':'sample ID for building a by-cell model'},
    {'name': 'by_drug',
        'type':str,
        'default':None,
        'help':'drug ID or name for building a by-drug model'},
# Data set selection
    {'name':'train_sources',
        'nargs':'+',
        'choices':['all', 'CCLE', 'CTRP', 'gCSI', 'GDSC', 'NCI60', 'SCL', 'SCLC', 'ALMANAC'],
        'help':'use one or more sources of drug response data for training'},
    {'name':'test_sources',
        'nargs':'+',
        'choices':['train', 'all', 'CCLE', 'CTRP', 'gCSI', 'GDSC', 'NCI60', 'SCL', 'SCLC', 'ALMANAC'],
        'help':'use one or more sources of drug response data for testing'},
# Sample selection
    {'name':'cell_types',
        'nargs':'+',
        'help':'limit training and test data to one or more tissue types'},
    {'name':'cell_subset_path',
        'type': str,
        'default': '',
        'help':'path for file with space delimited molecular sample IDs to keep'},
    {'name':'drug_subset_path',
        'type': str,
        'default': '',
        'help':'path for file with space delimited drug IDs to keep'},
    {'name':'drug_median_response_min',
        'type':float,
        'default':-1,
        'help':'keep drugs whose median response is greater than the threshold'},
    {'name':'drug_median_response_max',
        'type':float,
        'default':1,
        'help':'keep drugs whose median response is less than the threshold'},
# Training
    {'name':'no_feature_source',
        'type': candle.str2bool,
        'default': False,
        'help':'do not embed cell or drug feature source as part of input'},
    {'name':'no_response_source',
        'type': candle.str2bool,
        'default': False,
        'help':'do not encode response data source as an input feature'},
    {'name':'dense_feature_layers',
        'nargs':'+',
        'type':int,
        'help':'number of neurons in intermediate dense layers in the feature encoding submodels'},
    {'name':'use_landmark_genes',
        'type': candle.str2bool,
        'default': False,
        'help':'use the 978 landmark genes from LINCS (L1000) as expression features'},
    {'name':'use_filtered_genes',
        'type': candle.str2bool,
        'default': False,
        'help':'use the variance filtered genes as expression features'},
    {'name':'feature_subset_path',
        'type': str,
        'default': '',
        'help':'path for file with space delimited features to keep'},
    {'name':'cell_feature_subset_path',
        'type': str,
        'default': '',
        'help':'path for file with space delimited molecular features to keep'},
    {'name':'drug_feature_subset_path',
        'type': str,
        'default': '',
        'help':'path for file with space delimited drug features to keep'},
    {'name':'preprocess_rnaseq',
        'choices':['source_scale', 'combat', 'none'],
        'default':'none',
        'help':'preprocessing method for RNAseq data; none for global normalization'},
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
        'type':float,
        'default':None,
        'help':'base learning rate'},
    {'name':'cp',
        'type': candle.str2bool,
        'default': False,
        'help':'checkpoint models with best val_loss'},
    {'name':'tb',
        'type': candle.str2bool,
        'default': False,
        'help':'use tensorboard'},
    {'name': 'tb_prefix',
        'type': str,
        'default': 'tb',
        'help': 'prefix name for tb log'},
    {'name':'max_val_loss',
        'type':float,
        'default':argparse.SUPPRESS,
        'help':'retrain if val_loss is greater than the threshold'},
    {'name':'partition_by',
        'choices':['index', 'drug_pair', 'cell'],
        'default':None,
        'help':'cross validation paritioning scheme'},
     {'name':'cv',
        'type':int,
        'default':argparse.SUPPRESS,
        'help':'cross validation folds'},
    {'name':'no_gen',
        'type': candle.str2bool,
        'default': False,
        'help':'do not use generator for training and validation data'},
    {'name':'cache',
        'type': str,
        'default': None,
        'help':'prefix of data cache files to use'},
    {'name':'single',
        'type': candle.str2bool,
        'default': False,
        'help':'do not use drug pair representation'},
    {'name': 'export_csv',
        'type': str,
        'default': None,
        'help': 'output csv file name'},
    {'name':'export_data',
        'type': str,
        'default': None,
        'help':'output dataframe file name'},
    {'name': 'use_exported_data',
        'type': str,
        'default': None,
        'help': 'exported file name'},
     {'name':'growth_bins',
        'type': int,
        'default': 0,
        'help':'number of bins to use when discretizing growth response'},
    {'name' : 'initial_weights',
        'type' : str,
        'default': None,
        'help' : 'file name of initial weights'},
    {'name' : 'save_weights',
        'type': str,
        'default' : None,
        'help': 'name of file to save weights to' },
    {'name':'exclude_cells', 'nargs':'+',
        'default': [],
        'help':'cell line IDs to exclude'},
    {'name':'exclude_drugs', 'nargs':'+',
        'default': [],
        'help':'drug line IDs to exclude'},
    {'name':'sample_repetition',
        'type': candle.str2bool,
        'default': False,
        'help':'allow repetition of training data'}
]



required = [
    'activation',
    'batch_size',
    'dense',
    'dense_feature_layers',
    'drop',
    'epochs',
    'feature_subsample',
    'learning_rate',
    'loss',
    'optimizer',
    'residual',
    'rng_seed',
    'save_path',
    'scaling',
    'val_split',
    'solr_root',
    'timeout'
    ]
