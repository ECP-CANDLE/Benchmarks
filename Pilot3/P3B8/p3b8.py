import os

file_path = os.path.dirname(os.path.realpath(__file__))

import candle

additional_definitions = [
    {"name": "learning_rate_min", "action": "store", "type": float},
    {"name": "log_interval", "action": "store", "type": int},
    {"name": "weight_decay", "action": "store", "type": float},
    {"name": "grad_clip", "action": "store", "type": int},
    {"name": "unrolled", "action": "store", "type": candle.str2bool},
    {"name": "device", "action": "store", "type": str},
    {"name": "num_train_samples", "action": "store", "type": int},
    {"name": "num_valid_samples", "action": "store", "type": int},
    {"name": "num_test_samples", "action": "store", "type": int},
    {"name": "num_classes", "action": "store", "type": int},
    {"name": "eps", "action": "store", "type": float},
]

required = [
    "learning_rate",
    "weight_decay",
    "rng_seed",
    "batch_size",
    "epochs",
]


class BenchmarkP3B8(candle.Benchmark):
    """Benchmark for Quantized BERT"""

    def set_locals(self):
        """Set parameters for the benchmark.

        Args:
            required: set of required parameters for the benchmark.
            additional_definitions: list of dictionaries describing the additional parameters for the
            benchmark.
        """
        if required is not None:
            self.required = set(required)
        if additional_definitions is not None:
            self.additional_definitions = additional_definitions
