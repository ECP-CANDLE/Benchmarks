from __future__ import print_function

import numpy as np

from sklearn.metrics import accuracy_score

import os
import sys
import argparse

file_path = os.path.dirname(os.path.realpath(__file__))
lib_path2 = os.path.abspath(os.path.join(file_path, '..', '..', 'common'))
sys.path.append(lib_path2)

import candle



required = [
    'learning_rate', 'batch_size', 'epochs', 'dropout', \
    'optimizer', 'wv_len', \
    'attention_size']



class BenchmarkP3B3(candle.Benchmark):

    def set_locals(self):
        """Functionality to set variables specific for the benchmark
        - required: set of required parameters for the benchmark.
        - additional_definitions: list of dictionaries describing the additional parameters for the
        benchmark.
        """

        if required is not None:
            self.required = set(required)
        # if additional_definitions is not None:
            # self.additional_definitions = additional_definitions




