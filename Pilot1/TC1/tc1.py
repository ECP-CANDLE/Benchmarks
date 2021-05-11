from __future__ import print_function

import os
import sys
import logging

file_path = os.path.dirname(os.path.realpath(__file__))
lib_path2 = os.path.abspath(os.path.join(file_path, '..', '..', 'common'))
sys.path.append(lib_path2)

import candle

logger = logging.getLogger(__name__)

additional_definitions = [
    {'name': 'pool',
     'nargs': '+',
     'type': int,
     'help': 'network structure of shared layer'},
    {'name': 'classes',
     'type': int,
     'default': 36}
]

required = [
    'data_url',
    'train_data',
    'test_data',
    'model_name',
    'conv',
    'dense',
    'activation',
    'out_activation',
    'loss',
    'optimizer',
    'feature_subsample',
    'metrics',
    'epochs',
    'batch_size',
    'dropout',
    'classes',
    'pool',
    'output_dir'
]


class BenchmarkTC1(candle.Benchmark):

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


def load_data(params):

    train_path = candle.fetch_file(params['data_url'] + params['train_data'], 'Pilot1')
    test_path = candle.fetch_file(params['data_url'] + params['test_data'], 'Pilot1')

    if params['feature_subsample'] > 0:
        usecols = list(range(params['feature_subsample']))
    else:
        usecols = None

    return candle.load_Xy_data_noheader(train_path, test_path, params['classes'], usecols,
                                        scaling='maxabs', dtype=params['data_type'])
