from __future__ import print_function

import os
import sys

file_path = os.path.dirname(os.path.realpath(__file__))
lib_path2 = os.path.abspath(os.path.join(file_path, '..', '..', 'common'))
sys.path.append(lib_path2)

import candle

additional_definitions = [
    {'name': 'train_features',
     'action': 'store',
     'default': 'data/task0_0_train_feature.csv;data/task1_0_train_feature.csv;data/task2_0_train_feature.csv',
     'help': 'training feature data filenames'},
    {'name': 'train_truths',
     'action': 'store',
     'default': 'data/task0_0_train_label.csv;data/task1_0_train_label.csv;data/task2_0_train_label.csv',
     'help': 'training truth data filenames'},
    {'name': 'valid_features',
     'action': 'store',
     'default': 'data/task0_0_test_feature.csv;data/task1_0_test_feature.csv;data/task2_0_test_feature.csv',
     'help': 'validation feature data filenames'},
    {'name': 'valid_truths',
     'action': 'store',
     'default': 'data/task0_0_test_label.csv;data/task1_0_test_label.csv;data/task2_0_test_label.csv',
     'help': 'validation truth data filenames'},
    {'name': 'output_files',
     'action': 'store',
     'default': 'result0_0.csv;result1_0.csv;result2_0.csv',
     'help': 'output filename'},
    {'name': 'shared_nnet_spec',
     'nargs': '+',
     'type': int,
     'help': 'network structure of shared layer'},
    {'name': 'ind_nnet_spec',
     'action': 'list-of-lists',
     'help': 'network structure of task-specific layer'},
    {'name': 'case',
     'default': 'CenterZ',
     'choices': ['Full', 'Center', 'CenterZ'],
     'help': 'case classes'},
    {'name': 'fig',
     'type': candle.str2bool,
     'default': False,
     'help': 'Generate Prediction Figure'},
    {'name': 'feature_names',
     'nargs': '+',
     'type': str},
    {'name': 'n_fold',
     'action': 'store',
     'type': int},
    {'name': 'emb_l2',
     'action': 'store',
     'type': float},
    {'name': 'w_l2',
     'action': 'store',
     'type': float},
    {'name': 'wv_len',
     'action': 'store',
     'type': int},
    {'name': 'filter_sets',
     'nargs': '+',
     'type': int},
    {'name': 'filter_sizes',
     'nargs': '+',
     'type': int},
    {'name': 'num_filters',
     'nargs': '+',
     'type': int}
]


required = [
    'learning_rate', 'batch_size', 'epochs', 'dropout',
    'optimizer', 'wv_len',
    'filter_sizes', 'filter_sets', 'num_filters', 'emb_l2', 'w_l2']


class BenchmarkP3B3(candle.Benchmark):

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
