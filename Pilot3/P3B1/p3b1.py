from __future__ import print_function

import numpy as np

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
     'type': int}
]


required = ['learning_rate', 'batch_size', 'epochs', 'dropout',
            'activation', 'out_activation', 'loss', 'optimizer', 'metrics',
            'n_fold', 'scaling', 'initialization', 'shared_nnet_spec',
            'ind_nnet_spec', 'feature_names']


class BenchmarkP3B1(candle.Benchmark):

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


def build_data(nnet_spec_len, fold, data_path):
    """ Build feature sets to match the network topology
    """
    X_train = []
    Y_train = []

    X_test = []
    Y_test = []

    for i in range(nnet_spec_len):
        feature_train = np.genfromtxt(data_path + '/task' + str(i) + '_' + str(fold) + '_train_feature.csv', delimiter=',')
        label_train = np.genfromtxt(data_path + '/task' + str(i) + '_' + str(fold) + '_train_label.csv', delimiter=',')
        X_train.append(feature_train)
        Y_train.append(label_train)

        feature_test = np.genfromtxt(data_path + '/task' + str(i) + '_' + str(fold) + '_test_feature.csv', delimiter=',')
        label_test = np.genfromtxt(data_path + '/task' + str(i) + '_' + str(fold) + '_test_label.csv', delimiter=',')
        X_test.append(feature_test)
        Y_test.append(label_test)

    return X_train, Y_train, X_test, Y_test
