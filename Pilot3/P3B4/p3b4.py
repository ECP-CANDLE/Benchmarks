from __future__ import print_function

import os
import sys

file_path = os.path.dirname(os.path.realpath(__file__))
lib_path2 = os.path.abspath(os.path.join(file_path, '..', '..', 'common'))
sys.path.append(lib_path2)

import candle

additional_definitions = [
    {'name': 'attention_size',
     'action': 'store',
     'type': int},
    {'name': 'embed_train',
     'action': 'store',
     'type': candle.str2bool},
    {'name': 'min_lines',
     'action': 'store',
     'type': int},
    {'name': 'max_lines',
     'action': 'store',
     'type': int},
    {'name': 'min_words',
     'action': 'store',
     'type': int},
    {'name': 'max_words',
     'action': 'store',
     'type': int},
    {'name': 'wv_len',
     'action': 'store',
     'type': int}
]

required = [
    'learning_rate', 'batch_size', 'epochs', 'dropout',
    'optimizer', 'wv_len',
    'min_lines', 'max_lines',
    'min_words', 'max_words',
    'attention_size', 'embed_train'
]


class BenchmarkP3B3(candle.Benchmark):

    def set_locals(self):
        """Functionality to set variables specific for the benchmark
        - required: set of required parameters for the benchmark.
        - additional_definitions: list of dictionaries describing the
          additional parameters for the benchmark.
        """

        if required is not None:
            self.required = set(required)
        if additional_definitions is not None:
            self.additional_definitions = additional_definitions
