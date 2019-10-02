import candle
import p3b5 as bmk


required = [
    'learning_rate', 'batch_size', 'epochs', 'dropout', \
    'optimizer', 'wv_len', \
    'filter_sizes', 'filter_sets', 'num_filters', 'emb_l2', 'w_l2']


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
