import os
import tensorflow as tf
from tensorflow.keras import backend as K

def set_parallelism_threads():
    """ Set the number of parallel threads according to the number available on the hardware
    """

    if K.backend() == 'tensorflow' and 'NUM_INTRA_THREADS' in os.environ and 'NUM_INTER_THREADS' in os.environ:
        import tensorflow as tf
        # print('Using Thread Parallelism: {} NUM_INTRA_THREADS, {} NUM_INTER_THREADS'.format(os.environ['NUM_INTRA_THREADS'], os.environ['NUM_INTER_THREADS']))
        session_conf = tf.ConfigProto(inter_op_parallelism_threads=int(os.environ['NUM_INTER_THREADS']),
            intra_op_parallelism_threads=int(os.environ['NUM_INTRA_THREADS']))
        sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
        K.set_session(sess)

