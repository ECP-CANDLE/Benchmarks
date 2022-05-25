from __future__ import print_function

import os
import logging

file_path = os.path.dirname(os.path.realpath(__file__))

import candle

logger = logging.getLogger(__name__)
candle.set_parallelism_threads()

additional_definitions = [
    {'name': 'filename',
     'type': str,
     'help': 'name of tab-delimited file for the list of SMILES'},
    {'name': 'colname',
     'type': str,
     'help': 'column name for SMILES'}
]

required = ['filename']


class BenchmarkConvertSmiles(candle.Benchmark):

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
