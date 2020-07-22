from __future__ import print_function

import os
import sys

file_path = os.path.dirname(os.path.realpath(__file__))
lib_path2 = os.path.abspath(os.path.join(file_path, '..', '..', 'common'))
sys.path.append(lib_path2)

import candle

additional_definitions = [
    {'name': 'rnn_size',
     'action': 'store',
     'type': int,
     'help': 'size of LSTM internal state'},
    {'name': 'n_layers',
     'action': 'store',
     'help': 'number of layers in the LSTM'},
    {'name': 'do_sample',
     'type': candle.str2bool,
     'help': 'generate synthesized text'},
    {'name': 'temperature',
     'action': 'store',
     'type': float,
     'help': 'variability of text synthesis'},
    {'name': 'primetext',
     'action': 'store',
     'help': 'seed string of text synthesis'},
    {'name': 'length',
     'action': 'store',
     'type': int,
     'help': 'length of synthesized text'},
]

required = ['train_data', 'rnn_size', 'epochs', 'n_layers',
            'learning_rate', 'dropout', 'recurrent_dropout',
            'temperature', 'primetext', 'length']


class BenchmarkP3B2(candle.Benchmark):

    def set_locals(self):
        """Functionality to set variables specific for the benchmark
        - required: set of required parameters for the benchmark.
        - additional_definitions: list of dictionaries \
              describing the additional parameters for the
        benchmark.
        """

        if required is not None:
            self.required = set(required)
        if additional_definitions is not None:
            self.additional_definitions = additional_definitions
