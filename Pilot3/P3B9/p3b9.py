import os
import sys

file_path = os.path.dirname(os.path.realpath(__file__))
lib_path2 = os.path.abspath(os.path.join(file_path, '..', '..', 'common'))
sys.path.append(lib_path2)

import candle

additional_definitions = [
    {'name': 'per_device_train_batch_size',
        'type': int,
        'default': 16,
        'help': 'Batch per device in training'},
    {'name': 'gradient_accumulation_steps',
        'type': int,
        'default': 1,
        'help': 'Number of steps for accumulating gradient'},
    {'name': 'max_len',
        'type': int,
        'default': 512,
        'help': 'Max length for'},
    {'name': 'weight_decay',
        'type': float,
        'default': 0.0000,
        'help': 'ADAM weight decay'},
    {'name': 'adam_beta2',
        'type': float,
        'default': 0.98,
        'help': 'ADAM beta2 parameter'},
    {'name': 'adam_epsilon',
        'type': float,
        'default': 2e-8,
        'help': 'ADAM epsilon parameter'},
    {'name': 'max_steps',
        'type': int,
        'default': 10,
        'help': 'Max training steps'},
    {'name': 'warmup_steps',
        'type': int,
        'default': 1,
        'help': 'Warmup steps'},
    {'name': 'output_dir',
        'type': str,
        'default': './outputs/',
        'help': 'Output directory'},
]

required = [
    'output_dir',
    'train_bool',
    'per_device_train_batch_size',
    'gradient_accumulation_steps',
    'max_len',
    'learning_rate',
    'weight_decay',
    'adam_beta2',
    'adam_epsilon',
    'max_steps',
    'warmup_steps',
]


class BenchmarkP3B9(candle.Benchmark):
    """ Benchmark for BERT """

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
