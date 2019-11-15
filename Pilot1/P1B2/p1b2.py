from __future__ import print_function

import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score

import os
import sys
import logging
import argparse
try:
    import configparser
except ImportError:
    import ConfigParser as configparser


file_path = os.path.dirname(os.path.realpath(__file__))
lib_path = os.path.abspath(os.path.join(file_path, '..', 'common'))
sys.path.append(lib_path)
lib_path2 = os.path.abspath(os.path.join(file_path, '..', '..', 'common'))
sys.path.append(lib_path2)

import candle

#url_p1b2 = 'http://ftp.mcs.anl.gov/pub/candle/public/benchmarks/P1B2/'
#file_train = 'P1B2.train.csv'
#file_test = 'P1B2.test.csv'

logger = logging.getLogger(__name__)

additional_definitions = []

required = [
    'data_url',
    'train_data',
    'test_data',
    'activation',
    'batch_size',
    'dense',
    'drop',
    'epochs',
    'feature_subsample',
    'initialization',
    'learning_rate',
    'loss',
    'optimizer',
    'penalty',
    'rng_seed',
    'scaling',
    'validation_split',
    'shuffle'
]

class BenchmarkP1B2(candle.Benchmark):

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

def extension_from_parameters(params, framework):
    """Construct string for saving model with annotation of parameters"""
    ext = framework
    ext += '.A={}'.format(params['activation'])
    ext += '.B={}'.format(params['batch_size'])
    ext += '.D={}'.format(params['drop'])
    ext += '.E={}'.format(params['epochs'])
    if params['feature_subsample']:
        ext += '.F={}'.format(params['feature_subsample'])
    for i, n in enumerate(params['dense']):
        if n:
            ext += '.D{}={}'.format(i+1, n)
    ext += '.S={}'.format(params['scaling'])

    return ext


def load_data_one_hot(params, seed):
   # fetch data
    file_train = candle.fetch_file(params['data_url'] + params['train_data'],subdir='Pilot1')
    file_test = candle.fetch_file(params['data_url'] + params['test_data'],subdir='Pilot1')

    return candle.load_Xy_one_hot_data2(file_train, file_test, class_col=['cancer_type'],
                                           drop_cols=['case_id', 'cancer_type'],
                                           n_cols=params['feature_subsample'],
                                           shuffle=params['shuffle'],
                                           scaling=params['scaling'],
                                           validation_split=params['validation_split'],
                                           dtype=params['datatype'],
                                           seed=seed)


def load_data(params, seed):
   # fetch data
    file_train = candle.fetch_file(params['data_url'] + params['train_data'],subdir='Pilot1')
    file_test = candle.fetch_file(params['data_url'] + params['test_data'],subdir='Pilot1')

    return candle.load_Xy_data2(file_train, file_test, class_col=['cancer_type'],
                                  drop_cols=['case_id', 'cancer_type'],
                                  n_cols=params['feature_subsample'],
                                  shuffle=params['shuffle'],
                                  scaling=params['scaling'],
                                  validation_split=params['validation_split'],
                                  dtype=params['datatype'],
                                  seed=seed)


def evaluate_accuracy_one_hot(y_pred, y_test):
    def map_max_indices(nparray):
        maxi = lambda a: a.argmax()
        iter_to_na = lambda i: np.fromiter(i, dtype=np.float)
        return np.array([maxi(a) for a in nparray])
    ya, ypa = tuple(map(map_max_indices, (y_test, y_pred)))
    accuracy = accuracy_score(ya, ypa)
    # print('Accuracy: {}%'.format(100 * accuracy))
    return {'accuracy': accuracy}


def evaluate_accuracy(y_pred, y_test):
    accuracy = accuracy_score(y_test, y_pred)
    # print('Accuracy: {}%'.format(100 * accuracy))
    return {'accuracy': accuracy}
