import os
import sys
import candle


file_path = os.path.dirname(os.path.realpath(__file__))
lib_path2 = os.path.abspath(os.path.join(file_path, '..', '..', 'common'))
sys.path.append(lib_path2)


REQUIRED = [
    'learning_rate', 
    'learning_rate_min',
    'momentum',
    'weight_decay',
    'grad_clip',
    'seed',
    'unrolled',
    'batch_size', 
    'epochs', 
]


class BenchmarkP3B5(candle.Benchmark):
    """ Benchmark for P3B5 """

    def set_locals(self):
        """ Set parameters for the benchmark.

        Args: 
            required: set of required parameters for the benchmark.
        """
        if REQUIRED is not None:
            self.required = set(REQUIRED)
