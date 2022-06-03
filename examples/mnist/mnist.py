import os

file_path = os.path.dirname(os.path.realpath(__file__))

import candle
from tensorflow.keras import backend as K

# thread optimization
if K.backend() == "tensorflow" and "NUM_INTRA_THREADS" in os.environ:
    import tensorflow as tf

    sess = tf.Session(
        config=tf.ConfigProto(
            inter_op_parallelism_threads=int(os.environ["NUM_INTER_THREADS"]),
            intra_op_parallelism_threads=int(os.environ["NUM_INTRA_THREADS"]),
        )
    )
    K.set_session(sess)


additional_definitions = None
required = None


class MNIST(candle.Benchmark):
    def set_locals(self):
        if required is not None:
            self.required = set(required)
        if additional_definitions is not None:
            self.additional_definitions = additional_definitions
