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

from default_utils import set_up_logger
from default_utils import set_seed as set_seed_defaultUtils

import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    from sklearn.metrics import r2_score

import json
import os
import time

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

"""
Hyperparameters that affect CANDLE checkpoint/restart:

checksum: boolean: If True, use checksums on model.h5.  Default: True.
restart:  boolean: If True, automatically try to restart from model.h5.
                   Default: True

"""
            
class CandleCheckpointCallback(Callback):

    """
    Keras Callback for CANDLE-compliant Benchmarks to use for checkpointing
    Creates a JSON file alongside the weights and optimizer checkpoints
    that includes important metadata, particularly for restarting and
    tracking complex workflows.
    """

    def __init__(self, logger="DEFAULT",
                 save_best_only=True, save_weights_only=True,
                 save_best_stat=None,
                 skip_epochs=0,
                 checksum_model=False,
                 metadata=None, clean=True, verbose=True):
        """
        Parameters
        ----------
            logger : Logger
                The logger to use.
                May be None to disable or "DEFAULT" to use the default.
            save_best_only : boolean
                If true, only save when save_best_stat has improved.
            save_best_stat : string
                Required when save_best_only=True, else unused.
                The stat in logs.model to track for improvement.
            skip_epochs : int
                Number of initial epochs to skip before writing checkpoints
            checksum_model : boolean
                If True, compute a checksum for the model
                and store it in the JSON
            metadata : string
                Arbitrary string to add to the JSON file regarding
                job ID, hardware location, etc.
                May be None or an empty string.
            clean : boolean
                If True, remove old checkpoints immediately.
                If False, one extra old checkpoint will remain on disk.
            verbose : boolean
                If True, more verbose logging
                Passed to default_utils.set_up_logger(verbose) for this logger
        """
        import math
        self.logger = logger
        if self.logger == "DEFAULT":
            import logging
            self.logger = logging.getLogger("CandleCheckpointCallback")
            set_up_logger("save/ckpt.log", self.logger, verbose=verbose,
                          fmt_line="%(asctime)s CandleCheckpoint: %(message)s")
        self.skip_epochs = skip_epochs
        self.save_best_only = save_best_only
        self.save_best_stat = save_best_stat
        self.best_stat_last = math.inf
        self.save_weights_only = save_weights_only
        self.checksum_model = checksum_model
        self.metadata = metadata
        self.timestamp_last = None
        self.clean = clean
        self.info("Callback initialized.")
        if self.metadata is not None:
            self.info("metadata='%s'" % self.metadata)
        if self.save_best_only:
            self.info("save_best_stat='%s'" % self.save_best_stat)

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
        model_file = dir_work+"/model.h5"
        if not self.save_check(logs, epoch): return
        self.write_model(dir_work, epoch)
        import shutil
        if os.path.exists(dir_old):
            self.debug("removing: '%s'" % dir_old)
            shutil.rmtree(dir_old)
        do_clean = self.clean
        if os.path.exists(dir_good):
            self.debug("renaming: '%s' -> '%s'" % (dir_good, dir_old))
            os.rename(dir_good, dir_old)
        else:
            do_clean = False
        self.debug("renaming: '%s' -> '%s'" % (dir_work, dir_good))
        os.rename(dir_work, dir_good)
        if do_clean:
            self.debug("removing: '%s'" % dir_old)
            shutil.rmtree(dir_old)
            
    def save_check(self, logs, epoch):
        """
        Make sure we want to save this epoch based on the
        model metrics in given logs
        """
        # skip early epochs to improve speed 
        if epoch < self.skip_epochs:
            self.debug("Model saving disabled until epoch %d" % self.skip_epochs)
            return False
        if not self.save_best_only:
            return True # easy- save everything!
        if self.save_best_stat not in logs.keys():
            raise(Exception(("CandleCheckpointCallback: " +
                             "save_best_stat='%s' " +
                             "not in list of model metrics: %s") %
                            (self.save_best_stat, str(logs.keys()))))
        if   logs[self.save_best_stat] < self.best_stat_last:
            symbol =                  "<"
        elif logs[self.save_best_stat] > self.best_stat_last:
            symbol =                  ">"
        else:
            symbol =                  "="
        self.debug("metrics: current=%f %s last=%f" %
                   (logs[self.save_best_stat], symbol, self.best_stat_last))
        if logs[self.save_best_stat] < self.best_stat_last:
            self.best_stat_last = logs[self.save_best_stat]
            return True # model improved- save!
        # else- not saving:
        self.debug("not writing this epoch.")
        return False

    def write_model(self, dir_work, epoch):
        """ Do the I/O, report stats """
        model_file = dir_work + "/model.h5"
        self.debug("writing model to: '%s'" % model_file)
        start = time.time()
        self.model.save(model_file)  # save_format="h5")
        stop = time.time()
        duration = stop - start
        stats = os.stat(model_file)
        MB = stats.st_size / (1024*1024)
        rate = MB / duration
        self.debug("model wrote: %0.3f MB in %0.3f seconds (%0.2f MB/s)." %
                   (MB, duration, rate))
        self.checksum(dir_work)
        self.write_json(dir_work+"/ckpt-info.json", epoch)
    
    def checksum(self, dir_work):
        """ Simple checksum dispatch """
        if self.checksum_model:
            self.cksum_model = checksum_file(self.logger, dir_work+"/model.h5")
        else:
            self.cksum_model = "__DISABLED__"

    def write_json(self, jsonfile, epoch):
        from datetime import datetime
        now = datetime.now()
        D = {}
        D["epoch"] = epoch
        D["save_best_only"] = self.save_best_only
        D["save_best_stat"] = self.save_best_stat
        D["model_file"] = "model.h5"
        D["checksum"] = self.cksum_model
        D["timestamp"] = now.strftime("%Y-%m-%d %H:%M:%S")
        if self.timestamp_last == None:
            time_elapsed = "__FIRST__"
        else:
            time_elapsed = (now - self.timestamp_last).total_seconds()
        self.timestamp_last = now
        D["time_elapsed"] = time_elapsed
        D["metadata"] = self.metadata
        with open(jsonfile, "w") as fp:
            json.dump(D, fp)
            fp.write("\n")

    def info(self, message):
        if self.logger is not None:
            self.logger.info(message)

    def debug(self, message):
        if self.logger is not None:
            self.logger.debug(message)

def restart(gParameters, model, verbose=True):
    """
    Possibly restarts model from CheckpointCallback according to given
    settings and the ckpt-info.json

    return 
           The JSON dict if the restart happened or
           None if the restart did not happen.
    """
    import logging
    logger = logging.getLogger("Candle.restart")
    set_up_logger("save/ckpt.log", logger, verbose=verbose,
                  fmt_line="%(asctime)s CANDLE restart(): %(message)s")

    if disabled(gParameters, "restart"):
        return
    
    dir_work = "save/ckpt-work"
    dir_good = "save/ckpt-good"
    dir_old  = "save/ckpt-old"
    model_file = dir_good + "/model.h5"
    if not os.path.exists(model_file):
        return None
    logger.info("restarting: " + model_file)
    result = restart_json(gParameters, logger, dir_good)
    logger.info("restarting: epoch=%i timestamp=%s" %
                (result["epoch"], result["timestamp"]))
    start = time.time()
    stats = os.stat(model_file)
    MB = stats.st_size / (1024*1024)
    model.load_weights(model_file)
    stop = time.time()
    duration = stop - start
    rate = MB / duration
    logger.info("model read:  %0.3f MB in %0.3f seconds (%0.2f MB/s)." %
                (MB, duration, rate))
    return result

def restart_json(gParameters, logger, directory):
    json_file = directory + "/ckpt-info.json"
    if not os.path.exists(json_file):
        msg = "restart_json(): in: %s model exists but not json!" % \
              directory
        logger.info(msg)
        if not disabled(gParameters, "require_json"):
            raise Exception(msg)
    with open(json_file) as fp:
        J = json.load(fp)
    print(str(J))

    if not disabled(gParameters, "checksum"):
        checksum = checksum_file(logger, directory + "/model.h5")
        if checksum != J["checksum"]:
            raise Exception("checksum mismatch! directory: " %
                            directory)
    
    return J


def enabled(gParameters, key):
    """ Is this parameter set to True? """
    return key in gParameters and gParameters[key]


def disabled(gParameters, key):
    """ Is this parameter set to False? """
    return key in gParameters and not gParameters[key]

    
def checksum_file(logger, filename):
    """ Read file, compute checksum, return it as a string. """
    import zlib
    start = time.time()
    chunk_size = 10*1024*1024
    total = 0
    with open(filename, "rb") as fp:
        checksum = 0
        while True:
            chunk = fp.read(chunk_size)
            if not chunk:
                break
            total += len(chunk)
            checksum = zlib.crc32(chunk, checksum)
    stop = time.time()
    MB = total / (1024*1024)
    duration = stop - start
    rate = MB / duration
    logger.info("checksummed: %0.3f MB in %.3f seconds (%.2f MB/s)." %
                (MB, duration, rate))
    return str(checksum)
