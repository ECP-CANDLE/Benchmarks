from __future__ import absolute_import


from keras import backend as K
from keras import optimizers
from keras import initializers

from keras.layers import Dropout
from keras.callbacks import Callback, ModelCheckpoint
from keras.utils import get_custom_objects
from keras.metrics import binary_crossentropy, mean_squared_error, mean_absolute_error
from keras.models import Model

from scipy.stats.stats import pearsonr

from default_utils import set_seed as set_seed_defaultUtils

import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    from sklearn.metrics import r2_score

import os


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


def set_seed(seed):
    """ Set the random number seed to the desired value

        Parameters
        ----------
        seed : integer
            Random number seed.
    """

    set_seed_defaultUtils(seed)

    if K.backend() == 'tensorflow':
        import tensorflow as tf
        if tf.__version__ < "2.0.0":
            tf.set_random_seed(seed)
        else:
            tf.random.set_seed(seed)


def get_function(name):
    mapping = {}

    mapped = mapping.get(name)
    if not mapped:
        raise Exception('No keras function found for "{}"'.format(name))

    return mapped


def build_optimizer(type, lr, kerasDefaults):
    """ Set the optimizer to the appropriate Keras optimizer function
        based on the input string and learning rate. Other required values
        are set to the Keras default values

        Parameters
        ----------
        type : string
            String to choose the optimizer

            Options recognized: 'sgd', 'rmsprop', 'adagrad', adadelta', 'adam'
            See the Keras documentation for a full description of the options

        lr : float
            Learning rate

        kerasDefaults : list
            List of default parameter values to ensure consistency between frameworks

        Returns
        ----------
        The appropriate Keras optimizer function
    """

    if type == 'sgd':
        return optimizers.SGD(lr=lr, decay=kerasDefaults['decay_lr'],
                              momentum=kerasDefaults['momentum_sgd'],
                              nesterov=kerasDefaults['nesterov_sgd'])  # ,
# clipnorm=kerasDefaults['clipnorm'],
# clipvalue=kerasDefaults['clipvalue'])

    elif type == 'rmsprop':
        return optimizers.RMSprop(lr=lr, rho=kerasDefaults['rho'],
                                  epsilon=kerasDefaults['epsilon'],
                                  decay=kerasDefaults['decay_lr'])  # ,
# clipnorm=kerasDefaults['clipnorm'],
# clipvalue=kerasDefaults['clipvalue'])

    elif type == 'adagrad':
        return optimizers.Adagrad(lr=lr,
                                  epsilon=kerasDefaults['epsilon'],
                                  decay=kerasDefaults['decay_lr'])  # ,
# clipnorm=kerasDefaults['clipnorm'],
# clipvalue=kerasDefaults['clipvalue'])

    elif type == 'adadelta':
        return optimizers.Adadelta(lr=lr, rho=kerasDefaults['rho'],
                                   epsilon=kerasDefaults['epsilon'],
                                   decay=kerasDefaults['decay_lr'])  # ,
# clipnorm=kerasDefaults['clipnorm'],
# clipvalue=kerasDefaults['clipvalue'])

    elif type == 'adam':
        return optimizers.Adam(lr=lr, beta_1=kerasDefaults['beta_1'],
                               beta_2=kerasDefaults['beta_2'],
                               epsilon=kerasDefaults['epsilon'],
                               decay=kerasDefaults['decay_lr'])  # ,
# clipnorm=kerasDefaults['clipnorm'],
# clipvalue=kerasDefaults['clipvalue'])

# Not generally available
#    elif type == 'adamax':
#        return optimizers.Adamax(lr=lr, beta_1=kerasDefaults['beta_1'],
#                               beta_2=kerasDefaults['beta_2'],
#                               epsilon=kerasDefaults['epsilon'],
#                               decay=kerasDefaults['decay_lr'])

#    elif type == 'nadam':
#        return optimizers.Nadam(lr=lr, beta_1=kerasDefaults['beta_1'],
#                               beta_2=kerasDefaults['beta_2'],
#                               epsilon=kerasDefaults['epsilon'],
#                               schedule_decay=kerasDefaults['decay_schedule_lr'])


def build_initializer(type, kerasDefaults, seed=None, constant=0.):
    """ Set the initializer to the appropriate Keras initializer function
        based on the input string and learning rate. Other required values
        are set to the Keras default values

        Parameters
        ----------
        type : string
            String to choose the initializer

            Options recognized: 'constant', 'uniform', 'normal',
            'glorot_uniform', 'lecun_uniform', 'he_normal'

            See the Keras documentation for a full description of the options

        kerasDefaults : list
            List of default parameter values to ensure consistency between frameworks

        seed : integer
            Random number seed

        constant : float
            Constant value (for the constant initializer only)

        Return
        ----------
        The appropriate Keras initializer function
    """

    if type == 'constant':
        return initializers.Constant(value=constant)

    elif type == 'uniform':
        return initializers.RandomUniform(minval=kerasDefaults['minval_uniform'],
                                          maxval=kerasDefaults['maxval_uniform'],
                                          seed=seed)

    elif type == 'normal':
        return initializers.RandomNormal(mean=kerasDefaults['mean_normal'],
                                         stddev=kerasDefaults['stddev_normal'],
                                         seed=seed)

    elif type == 'glorot_normal':
        # aka Xavier normal initializer. keras default
        return initializers.glorot_normal(seed=seed)

    elif type == 'glorot_uniform':
        return initializers.glorot_uniform(seed=seed)

    elif type == 'lecun_uniform':
        return initializers.lecun_uniform(seed=seed)

    elif type == 'he_normal':
        return initializers.he_normal(seed=seed)


def xent(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred)


def r2(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return (1 - SS_res / (SS_tot + K.epsilon()))


def mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)


def mse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred)


def covariance(x, y):
    return K.mean(x * y) - K.mean(x) * K.mean(y)


def corr(y_true, y_pred):
    cov = covariance(y_true, y_pred)
    var1 = covariance(y_true, y_true)
    var2 = covariance(y_pred, y_pred)
    return cov / (K.sqrt(var1 * var2) + K.epsilon())


def evaluate_autoencoder(y_pred, y_test):
    mse = mean_squared_error(y_pred, y_test)
    r2 = r2_score(y_test, y_pred)
    corr, _ = pearsonr(y_pred.flatten(), y_test.flatten())
    # print('Mean squared error: {}%'.format(mse))
    return {'mse': mse, 'r2_score': r2, 'correlation': corr}


class PermanentDropout(Dropout):
    def __init__(self, rate, **kwargs):
        super(PermanentDropout, self).__init__(rate, **kwargs)
        self.uses_learning_phase = False

    def call(self, x, mask=None):
        if 0. < self.rate < 1.:
            noise_shape = self._get_noise_shape(x)
            x = K.dropout(x, self.rate, noise_shape)
        return x


def register_permanent_dropout():
    get_custom_objects()['PermanentDropout'] = PermanentDropout


class LoggingCallback(Callback):
    def __init__(self, print_fcn=print):
        Callback.__init__(self)
        self.print_fcn = print_fcn

    def on_epoch_end(self, epoch, logs={}):
        msg = "[Epoch: %i] %s" % (epoch, ", ".join("%s: %f" % (k, v) for k, v in sorted(logs.items())))
        self.print_fcn(msg)


class MultiGPUCheckpoint(ModelCheckpoint):

    def set_model(self, model):
        if isinstance(model.layers[-2], Model):
            self.model = model.layers[-2]
        else:
            self.model = model

class CandleCheckpointCallback(Callback):

    """
    Keras Callback for CANDLE-compliant Benchmarks to use for checkpointing
    Creates a JSON file alongside the weights and optimizer checkpoints
    that includes important metadata, particularly for restarting and
    tracking complex workflows.
    """

    def __init__(self, model_file, optimizer_file=None, logger=None,
                 save_best_only=True, save_weights_only=True,
                 save_best_stat=None,
                 checksum_model=False, checksum_optimizer=False,
                 metadata=None, clean=True):
        """
        Parameters
        ----------
            model_file : string
                Main model weights checkpoint file.
                Must be a writable file path.
            optimizer_file : string
                Checkpoint file for optimizer state.
                May be None to disable.
            logger : Logger
                The logger to use.
                May be None to disable.
            save_best_only : boolean
                If true, only save when save_best_stat has improved.
            save_best_stat : string
                Required when save_best_only=True, else unused.
                The stat in logs.model to track for improvement.
            checksum_model : boolean
                If True, compute a checksum for the model
                and store it in the JSON
            checksum_optimizer : boolean
                If True, compute a checksum for the optimizer
                and store it in the JSON
            metadata : string
                Arbitrary string to add to the JSON file regarding
                job ID, hardware location, etc.
                May be None or an empty string.
            clean : boolean
                If True, remove old checkpoints immediately.
                If False, one extra old checkpoint will remain on disk.
        """
        self.model_file = model_file
        self.optimizer_file = optimizer_file
        self.logger = logger
        self.save_best_only = save_best_only
        self.save_best_stat = save_best_stat
        self.save_weights_only = save_weights_only
        self.checksum_model = checksum_model
        self.checksum_optimizer = checksum_optimizer
        self.metadata = metadata
        self.timestamp_last = None
        self.clean = clean

    def on_epoch_end(self, epoch, logs):
        """
        Normally, ckpt-good is the best saved state.
        When updating:
        1. Write current state to ckpt-work
        2. Rename ckpt-good to ckpt-old
        3. Rename ckpt-work to ckpt-good
        4. Delete ckpt-old
        """
        # print("logs: %s" % str(logs.keys()))
        # TODO: Check save_best_only
        dir_work = "save/ckpt-work"
        dir_good = "save/ckpt-good"
        dir_old  = "save/ckpt-old"
        if not os.path.exists(dir_work):
            os.makedirs(dir_work)
        self.model.save(dir_work+"/model.h5", save_format="h5")
        if self.optimizer_file is not None:
            pass # TODO: optimizer_save()
        self.checksums(dir_work)
        self.write_json(dir_work+"/ckpt-info.json", epoch)
        import shutil
        if os.path.exists(dir_old):
            shutil.rmtree(dir_old)
        do_clean = self.clean
        if os.path.exists(dir_good):
            os.rename(dir_good, dir_old)
        else:
            do_clean = False
        os.rename(dir_work, dir_good)
        if do_clean:
            shutil.rmtree(dir_old)

    def checksums(self, dir_work):
        """ Simple checksum dispatch """
        if self.checksum_model:
            self.cksum_model = \
                self.checksum_file(dir_work+"/model.h5")
        else:
            self.cksum_model = "__DISABLED__"
        if self.checksum_optimizer:
            self.cksum_optimizer = \
                self.checksum_file(dir_work+"/optimizer.h5")
        else:
            self.cksum_optimizer = "__DISABLED__"

    def checksum_file(self, filename):
        """ Read file, compute checksum, return it. """
        import zlib
        chunk_size = 10*1024*1024
        with open(filename, "rb") as fp:
            checksum = 0
            while True:
                chunk = fp.read(chunk_size)
                if not chunk:
                    break
                checksum = zlib.crc32(chunk, checksum)
        return str(checksum)

    def write_json(self, jsonfile, epoch):
        from datetime import datetime
        now = datetime.now()
        D = {}
        D["epoch"] = epoch
        D["save_best_only"] = self.save_best_only
        D["save_best_stat"] = self.save_best_stat
        D["model_file"] = "model.h5"
        D["optimizer_file"] = "optimizer.h5"
        D["checksum_model"] = self.cksum_model
        D["checksum_optimizer"] = self.cksum_optimizer
        D["timestamp"] = now.strftime("%Y-%m-%d %H:%M:%S")
        if self.timestamp_last == None:
            time_elapsed = "__FIRST__"
        else:
            time_elapsed = (now - self.timestamp_last).total_seconds()
        self.timestamp_last = now
        D["time_elapsed"] = time_elapsed
        D["metadata"] = self.metadata
        import json
        with open(jsonfile, "w") as fp:
            json.dump(D, fp)
            fp.write("\n")
