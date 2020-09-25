import os
import sys

file_path = os.path.dirname(os.path.realpath(__file__))
lib_path2 = os.path.abspath(os.path.join(file_path, '..', '..', 'common'))
sys.path.append(lib_path2)

import candle

additional_definitions = [
{'name':'learning_rate_min',
    'action':'store',
    'type':float},
{'name':'log_interval',
    'action':'store',
    'type':int},
{'name':'weight_decay',
    'action':'store',
    'type':float},
{'name':'grad_clip',
    'action':'store',
    'type':int},
{'name':'unrolled',
    'action':'store',
    'type':candle.str2bool},
]

required = [
    'learning_rate',
    'weight_decay',
    'rng_seed',
    'batch_size',
    'num_epochs',
]


class BenchmarkP3B6(candle.Benchmark):
    """ Benchmark for P3B6 """

    def set_locals(self):
        """ Set parameters for the benchmark.

        Args:
            required: set of required parameters for the benchmark.
            additional_definitions: list of dictionaries describing the additional parameters for the
            benchmark.
        """
        if required is not None:
            self.required = set(required)
        if additional_definitions is not None:
                    self.additional_definitions = additional_definitions

