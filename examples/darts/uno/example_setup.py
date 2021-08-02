import os
import sys


file_path = os.path.dirname(os.path.realpath(__file__))
lib_path2 = os.path.abspath(os.path.join(file_path, '..', '..', '..', 'common'))
sys.path.append(lib_path2)


import candle

additional_definitions = [
    {'name': 'grad_clip', 'type': int},
    {'name': 'learning_rate_min', 'type': float, 'help': 'Minimum learning rate'},
    {'name': 'log_interval', 'type': int, 'help': 'Logging interval'},
    {'name': 'unrolled', 'type': candle.str2bool},
    {'name': 'weight_decay', 'type': float},
    {'name': 'grad_clip', 'type': int}
]

REQUIRED = [
    'learning_rate',
    'learning_rate_min',
    'momentum',
    'weight_decay',
    'grad_clip',
    'rng_seed',
    'batch_size',
    'epochs',
]


class UnoExample(candle.Benchmark):
    """ Example for Uno """

    def set_locals(self):
        """ Set parameters for the benchmark.

        Args:
        - required: set of required parameters for the benchmark.
        - additional_definitions: list of dictionaries describing the additional parameters for the
          benchmark.
        """
        if REQUIRED is not None:
            self.required = set(REQUIRED)
        if additional_definitions is not None:
            self.additional_definitions = additional_definitions
