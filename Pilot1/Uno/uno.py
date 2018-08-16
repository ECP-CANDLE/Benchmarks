from __future__ import print_function

import os
import sys
import logging
import argparse
try:
    import configparser
except ImportError:
    import ConfigParser as configparser


file_path = os.path.dirname(os.path.realpath(__file__))
lib_path = os.path.abspath(os.path.join(file_path, '..'))
sys.path.append(lib_path)
lib_path2 = os.path.abspath(os.path.join(file_path, '..', '..', 'common'))
sys.path.append(lib_path2)

import candle_keras as candle

logger = logging.getLogger(__name__)


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
        'type':str,
        'default':'argparse.SUPPRESS',
        'help':'limit training and test data to one or more tissue types'},
    {'name':'drug_median_response_min',
        'type':float,
        'default':-1,
        'help':'keep drugs whose median response is greater than the threshold'},
    {'name':'drug_median_response_max',
        'type':float,
        'default':1,
        'help':'keep drugs whose median response is less than the threshold'},
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
        #'default':'argparse.SUPPRESS',
        'help':'number of neurons in intermediate dense layers in the feature encoding submodels'},
    {'name':'use_landmark_genes',
        'type': candle.str2bool, 
        'default': False, 
        'help':'use the 978 landmark genes from LINCS (L1000) as expression features'},
    {'name':'use_filtered_genes',
        'type': candle.str2bool, 
        'default': False, 
        'help':'use the variance filtered genes as expression features'},
    {'name':'preprocess_rnaseq',
        'choices':['source_scale', 'combat', 'none'],
        'default':'none',
        'help':'preprocessing method for RNAseq data; none for global normalization'},
    {'name':'residual',
        'type': candle.str2bool, 
        'default': False, 
        'help':'add skip connections to the layers'},
# Training
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
    {'name':'export_data',
        'type': str,
        'default': None,
        'help':'output dataframe file name'},
     {'name':'growth_bins',
        'type':int,
        'default':0,
        'help':'number of bins to use when discretizing growth response'}
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
    'save',
    'scaling',
    'validation_split',
    'solr_root',
    'timeout'
    ]
