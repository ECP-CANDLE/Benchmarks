"""
CKPT KERAS UTILS

CANDLE checkpoint/restart utilities for Keras

Hyperparameters that affect CANDLE checkpoint/restart:

ckpt_restart_mode :  "off" | "auto" | "required"
    If 'auto' or 'required', automatically try to restart from most recent
    (highest epoch) model.h5.
    'required' will fail if a model cannot be found.
    Default: "auto"

ckpt_save_best : boolean
    If True, save whenever save_best_metric has improved.
    Default: True

ckpt_save_best_metric : string
    Required when ckpt_save_best=True, else unused.
    The metric in logs.model to track for improvement.
    Default: "val_loss"

ckpt_skip_epochs : integer
    Number of initial epochs to skip before writing checkpoints
    Default: 0

ckpt_save_interval: integer
    Save whenever epoch % save_interval == 0.
    Set save_interval=0 to disable these checkpoints (save nothing).
    Default: 1 (save everything)

ckpt_checksum : boolean
    If True, compute a checksum for the model
    and store it in the JSON.
    Also, confirm checksum at restart time.
    Default: False

ckpt_keep_mode : string
    "linear" or "exponential" (NYI)
    Default: "linear"

ckpt_keep_limit: integer GZ
    Maximal number of checkpoints to keep.
    This can be set lower to reduce disk usage.
    Default: 1000000

ckpt_directory: string
    The top directory to use.
    Default: "./save"
    Typical user values:
    "/tmp/user/ckpts": I.e. I am going to move these myself.
    "/other-fs/user/ckpts": I.e. My working FS is different from the FS
                            I want to use for checkpoints.

ckpt_metadata : string
    Arbitrary string to add to the JSON file regarding
    job ID, hardware location, etc.
    May be None or an empty string.
    Default: None

Usage:

  Add before training:

    initial_epoch = 0
    J = candle.restart(gParameters, model)
    if J is not None:
        initial_epoch = J['epoch']

  Set up a callback for checkpoints:

    ckpt = candle.CandleCheckpointCallback(gParameters)
    history = model.fit(epochs=gParameters['epochs'],
                        initial_epoch=initial_epoch,
                        ...
                        callbacks=[... , ckpt])

  Optionally, log a final report:

    ckpt.report_final()

Controlling restart:

Most restart control options are in gParameters.

Normally, restart() looks at the soft link ckpts/last,
which should point to a good ckpt directory number in epochs/*
and restarts from there.

To roll back, simply re-link ckpts/last to point to
a different directory number in epochs/* .
Any later epochs will simply be overwritten
(and a debug message will be reported).

If ckpts/last is missing or a broken link,
restart() will start from scratch.

Keep policy:

The ckpt_keep settings only apply to the current run.
Checkpoints from prior runs will never be deleted by clean().
You may simply remove any of them.
Normally you will not want to remove the one pointed to by ckpts/last,
but if you do, restart() will simply start from scratch.

Logging:

A log of ckpt operations is in ckpt_directory/ckpt.log

"""

import json
import os
import shutil
import time

from pathlib import PosixPath

from helper_utils import set_up_logger, str2bool
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import Callback, ModelCheckpoint


class MultiGPUCheckpoint(ModelCheckpoint):

    def set_model(self, model):
        if isinstance(model.layers[-2], Model):
            self.model = model.layers[-2]
        else:
            self.model = model


class ParamRequired:
    """ Indicates that the user params must contain this key """
    pass


class CandleCheckpointCallback(Callback):

    """
    Keras Callback for CANDLE-compliant Benchmarks to use for checkpointing
    Creates a JSON file alongside the weights and optimizer checkpoints
    that includes important metadata, particularly for restarting and
    tracking complex workflows.
    """

    def __init__(self, gParameters, logger="DEFAULT", verbose=True):
        """
        Parameters
        ----------
            logger : Logger
                The logger to use.
                May be None to disable or "DEFAULT" to use the default.
            verbose : boolean
                If True, more verbose logging
                Passed to helper_utils.set_up_logger(verbose) for this logger
        """
        self.logger = logger
        if self.logger == "DEFAULT":
            import logging
            self.logger = logging.getLogger("CandleCheckpointCallback")
            set_up_logger("save/ckpt.log", self.logger, verbose=verbose,
                          fmt_line="%(asctime)s CandleCheckpoint: %(message)s")
        self.scan_params(gParameters)
        # List of epoch integers this instance has written.
        # Sorted from smallest to largest.
        self.epochs = []
        # The best epoch wrt metric.  Do not delete this!
        self.epoch_best = 0
        self.report_initial()

    def report_initial(self):
        """ Simply report that we are ready to run """
        self.info("Callback initialized.")
        if self.metadata is not None:
            self.info("metadata='%s'" % self.metadata)
        if self.save_best:
            self.info("save_best_metric='%s'" % self.save_best_metric)
        self.info("PWD: " + os.getcwd())
        self.info("ckpt_directory: %s" %
                  PosixPath(self.ckpt_directory).resolve())

    def scan_params(self, gParameters):
        """ Simply translate gParameters into instance fields """
        self.epoch_max = param(gParameters, "epochs",
                               ParamRequired(), ParamType.INTEGER_NN)
        self.skip_epochs = param(gParameters, "ckpt_skip_epochs",
                                 0, ParamType.INTEGER_NN)
        self.ckpt_directory = param(gParameters, "ckpt_directory",
                                    "./save", ParamType.STRING)
        self.save_best = param(gParameters, "ckpt_save_best",
                               True, ParamType.BOOLEAN)
        self.save_best_metric = param(gParameters, "ckpt_save_best_metric",
                                      None, ParamType.STRING)
        self.best_metric_last = param(gParameters, "ckpt_best_metric_last",
                                      None, ParamType.FLOAT)
        if self.best_metric_last is None:
            import math
            self.best_metric_last = math.inf
        self.save_interval = param(gParameters, "ckpt_save_interval",
                                   1, ParamType.INTEGER_NN)
        self.save_weights_only = param(gParameters, "ckpt_save_weights_only",
                                       True, ParamType.BOOLEAN)
        self.checksum_enabled = param(gParameters, "ckpt_checksum",
                                      False, ParamType.BOOLEAN)
        self.keep_mode = param(gParameters, "ckpt_keep_mode",
                               "linear", ParamType.STRING,
                               allowed=[None, "all", "linear"])
        self.keep_limit = param(gParameters, "ckpt_keep_limit",
                                1000000, ParamType.INTEGER_GZ)
        self.metadata = param(gParameters, "metadata",
                              None, ParamType.STRING)
        self.timestamp_last = param(gParameters, "ckpt_timestamp_last",
                                    None, ParamType.STRING)
        self.cwd = os.getcwd()

    def on_epoch_end(self, epoch, logs=None):
        """
        Note: We immediately increment epoch
        from index-from-0 to index-from-1
        to match the TensorFlow output.
        Normally, ckpts/best is the best saved state,
              and ckpts/last is the last saved state.
        Procedure:
        1. Write current state to ckpts/work
        2. Rename ckpts/work to ckpts/epoch/NNN
        3. If best, link ckpts/best to ckpts/epoch/NNN
        4. Link ckpts/last to ckpts/epoch/NNN
        5. Clean up old ckpts according to keep policy
        """

        epoch += 1

        dir_root = PosixPath(self.ckpt_directory).resolve()
        dir_work = dir_root / "ckpts/work"
        dir_best = dir_root / "ckpts/best"  # a soft link
        dir_last = dir_root / "ckpts/last"  # a soft link
        dir_epochs = dir_root / "ckpts/epochs"
        dir_this = dir_epochs / ("%03i" % epoch)

        if not self.save_check(logs, epoch):
            return
        if os.path.exists(dir_this):
            self.debug("remove:  '%s'" % self.relpath(dir_this))
            shutil.rmtree(dir_this)
        os.makedirs(dir_epochs, exist_ok=True)
        os.makedirs(dir_work, exist_ok=True)
        self.write_model(dir_work, epoch)
        self.debug("rename:  '%s' -> '%s'" %
                   (self.relpath(dir_work), self.relpath(dir_this)))
        os.rename(dir_work, dir_this)
        self.epochs.append(epoch)
        if self.epoch_best == epoch:
            self.symlink(dir_this, dir_best)
        self.symlink(dir_this, dir_last)
        self.clean(epoch)

    def save_check(self, logs, epoch):
        """
        Make sure we want to save this epoch based on the
        model metrics in given logs
        Also updates epoch_best if appropriate
        """
        # skip early epochs to improve speed
        if epoch < self.skip_epochs:
            self.debug("model saving disabled until epoch %d" %
                       self.skip_epochs)
            return False
        # Do this first- it may set epoch_best:
        if self.save_check_best(logs, epoch):
            # The model improved- save!
            self.epoch_best = epoch
            return True
        if epoch == self.epoch_max:
            self.info("writing final epoch %i ..." % epoch)
            return True  # Final epoch - save!
        if self.save_interval != 0 and epoch % self.save_interval == 0:
            return True  # We are on the save_interval: save!
        # else- not saving:
        self.debug("not writing this epoch.")
        return False

    def save_check_best(self, logs, epoch):
        if not self.save_best:
            return False
        if self.save_best_metric not in logs.keys():
            raise Exception(("CandleCheckpointCallback: "
                             + "save_best_metric='%s' "
                             + "not in list of model metrics: %s") %
                            (self.save_best_metric, str(logs.keys())))

        # Known metrics and direction of progress
        known_metrics = {"loss": "-",
                         "accuracy": "+",
                         "val_loss": "-",
                         "val_accuracy": "+",
                         "lr": "-"}

        if self.save_best_metric not in known_metrics.keys():
            raise Exception(("CandleCheckpointCallback: "
                             + "save_best_metric='%s' "
                             + "not in list of known_metrics: %s") %
                            (self.save_best_metric,
                             str(known_metrics.keys())))

        # Logging:
        if logs[self.save_best_metric] < self.best_metric_last:
            symbol = "<"
        elif logs[self.save_best_metric] > self.best_metric_last:
            symbol = ">"
        else:
            symbol = "="
        self.debug("metrics: %s: current=%f %s last=%f" %
                   (self.save_best_metric,
                    logs[self.save_best_metric],
                    symbol, self.best_metric_last))

        # Check for improvement:
        improved = False  # did the metric improve this epoch?
        if known_metrics[self.save_best_metric] == "-":
            if logs[self.save_best_metric] < self.best_metric_last:
                improved = True
        elif known_metrics[self.save_best_metric] == "+":
            if logs[self.save_best_metric] > self.best_metric_last:
                improved = True
        else:
            assert(False)
        if improved:
            self.best_metric_last = logs[self.save_best_metric]
            self.epoch_best = epoch
            return True
        return False

    def write_model(self, dir_work, epoch):
        """
        Do the I/O, report stats
        dir_work: A PosixPath
        """
        model_file = dir_work / "model.h5"
        self.debug("writing model to: '%s'" % self.relpath(model_file))
        start = time.time()
        # TODO: Implement save_weights_only
        self.model.save(model_file)  # save_format="h5"
        stop = time.time()
        duration = stop - start
        stats = os.stat(model_file)
        MB = stats.st_size / (1024 * 1024)
        rate = MB / duration
        self.debug("model wrote: %0.3f MB in %0.3f seconds (%0.2f MB/s)." %
                   (MB, duration, rate))
        self.checksum(dir_work)
        self.write_json(dir_work / "ckpt-info.json", epoch)

    def checksum(self, dir_work):
        """
        Simple checksum dispatch
        dir_work: A PosixPath
        """
        if self.checksum_enabled:
            self.cksum_model = checksum_file(self.logger,
                                             dir_work / "model.h5")
        else:
            self.cksum_model = "__DISABLED__"

    def write_json(self, jsonfile, epoch):
        from datetime import datetime
        now = datetime.now()
        D = {}
        D["epoch"] = epoch
        D["save_best_metric"] = self.save_best_metric
        D["best_metric_last"] = self.best_metric_last
        D["model_file"] = "model.h5"
        D["checksum"] = self.cksum_model
        D["timestamp"] = now.strftime("%Y-%m-%d %H:%M:%S")
        if self.timestamp_last is None:
            time_elapsed = "__FIRST__"
        else:
            time_elapsed = (now - self.timestamp_last).total_seconds()
        self.timestamp_last = now
        D["time_elapsed"] = time_elapsed
        D["metadata"] = self.metadata
        with open(jsonfile, "w") as fp:
            json.dump(D, fp)
            fp.write("\n")

    def clean(self, epoch_now):
        """
        Clean old epoch directories
              in accordance with ckpt_keep policies.
        Return number of checkpoints kept and deleted
        """
        deleted = 0
        kept = 0
        # Consider most recent epochs first:
        for epoch in reversed(self.epochs):
            self.debug('checking %s' % epoch)
            if not self.keep(epoch, epoch_now, kept):
                deleted += 1
                self.delete(epoch)
                self.debug('deleted')
            else:
                kept += 1
                self.debug('kept %s' % kept)
        return (kept, deleted)

    def keep(self, epoch, epoch_now, kept):
        """
        kept: Number of epochs already kept
        return True if we are keeping this epoch
        """
        if epoch == epoch_now:
            # We just wrote this!
            self.debug('latest')
            return True
        if self.epoch_best == epoch:
            # This is the best epoch
            self.debug('best')
            return True
        if kept < self.keep_limit:
            self.debug('< limit %s' % self.keep_limit)
            return True
        # No reason to keep this: delete it:
        return False

    def delete(self, epoch):
        dir_old = "save/ckpts/epochs/%03i" % epoch
        if os.path.exists(dir_old):
            self.debug("removing: '%s'" % dir_old)
            shutil.rmtree(dir_old)
        else:
            self.info("checkpoint for epoch=%i disappeared!" %
                      epoch)
        self.epochs.remove(epoch)

    def symlink(self, src, dst):
        """ Like os.symlink, but overwrites dst and logs """
        self.debug("linking: '%s' -> '%s'" %
                   (self.relpath(dst), self.relpath(src)))
        if os.path.lexists(dst):
            os.remove(dst)
        os.symlink(src, dst)

    def relpath(self, p):
        return p.relative_to(self.cwd)

    def info(self, message):
        if self.logger is not None:
            self.logger.info(message)

    def debug(self, message):
        if self.logger is not None:
            self.logger.debug(message)

    def on_train_end(self, logs=None):
        self.report_final()

    def report_final(self):
        self.info("checkpoints kept: %i" %
                  len(self.epochs))
        self.info("checkpoints list: %s" %
                  str(self.epochs))


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
    directory = param(gParameters, "ckpt_directory", "./save")
    set_up_logger(directory + "/ckpt.log", logger,
                  verbose=verbose,
                  fmt_line="%(asctime)s CANDLE restart(): %(message)s")

    param_ckpt_mode = param(gParameters, "ckpt_restart_mode", "auto",
                            allowed=["off", "auto", "required"])
    if param_ckpt_mode == "off":
        return None

    dir_last = "save/ckpts/last"
    model_file = dir_last + "/model.h5"
    if not os.path.exists(model_file):
        if param_ckpt_mode == "required":
            raise Exception("ckpt_mode=='required' but no checkpoint"
                            + "could be found!")
        # We must be under AUTO - proceed without restart
        assert param_ckpt_mode == "auto"
        return None
    logger.info("restarting: %s", model_file)
    result = restart_json(gParameters, logger, dir_last)
    logger.info("restarting: epoch=%i timestamp=%s",
                result["epoch"], result["timestamp"])
    start = time.time()
    stats = os.stat(model_file)
    MB = stats.st_size / (1024 * 1024)
    model.load_weights(model_file)
    stop = time.time()
    duration = stop - start
    rate = MB / duration
    logger.info("model read:  %0.3f MB in %0.3f seconds (%0.2f MB/s).",
                MB, duration, rate)
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
    # print(str(J))
    logger.debug("ckpt-info.json contains:")
    logger.debug(json.dumps(J, indent=2))
    if param(gParameters, "ckpt_checksum", False, ParamType.BOOLEAN):
        checksum = checksum_file(logger, directory + "/model.h5")
        if checksum != J["checksum"]:
            raise Exception("checksum mismatch! directory: " %
                            directory)

    return J


from enum import Enum, unique, auto


@unique
class ParamType(Enum):
    """ Possible gParameters types """
    STRING = auto()
    BOOLEAN = auto()
    INTEGER = auto()
    # integer: non-negative
    INTEGER_NN = auto()
    # integer: greater-than-zero
    INTEGER_GZ = auto()
    FLOAT = auto()
    FLOAT_NN = auto()


def enabled(gParameters, key):
    """ Is this parameter set to True? """
    return key in gParameters and gParameters[key]


def disabled(gParameters, key):
    """ Is this parameter set to False? """
    return key in gParameters and not gParameters[key]


def param(gParameters, key, dflt,
          type_=ParamType.STRING, allowed=None):
    """ Pull key from parameters with type checks and conversions """
    if key in gParameters:
        result = gParameters[key]
    else:
        if isinstance(dflt, ParamRequired):
            raise Exception("param key must be provided: '%s'" % key)
        result = dflt
    result = param_type_check(key, result, type_)
    param_allowed(key, result, allowed)
    return result


def param_type_check(key, value, type_):
    """
    Check that value is convertable to given type:
          if not, raise TypeError
    Return the value as converted to given type
    """
    if value is None:
        return value
    if type_ is ParamType.STRING:
        return str(value)
    if type_ is ParamType.BOOLEAN:
        return param_type_check_bool(key, value)
    if type_ is ParamType.INTEGER or \
       type_ is ParamType.INTEGER_NN or \
       type_ is ParamType.INTEGER_GZ:
        return param_type_check_int(key, value, type_)
    if type_ is ParamType.FLOAT or \
       type_ is ParamType.FLOAT_NN:
        return param_type_check_float(key, value, type_)
    raise ValueError("param_type_check(): unknown type: '%s'" %
                     str(type_))


def param_type_check_bool(key, value):
    if isinstance(value, bool):
        return value
    try:
        v = str2bool(value)
    except TypeError:
        raise TypeError("parameter: '%s' is '%s' but must be a %s" %
                        key, str(value), str(ParamType.BOOLEAN))
    return v


def param_type_check_int(key, value, type_):
    if isinstance(value, int):
        result = value
    else:
        try:
            result = int(value)
        except TypeError:
            raise TypeError("parameter: '%s' is '%s' but must be a %s" %
                            (key, str(value), str(type_)))
    if type_ == ParamType.INTEGER_NN:
        if result < 0:
            raise TypeError(("parameter: '%s' is '%s' "
                             + "but must be non-negative") %
                            (key, str(value)))
    if type_ == ParamType.INTEGER_GZ:
        if result <= 0:
            raise TypeError(("parameter: '%s' is '%s' "
                             + "but must be greater-than-zero") %
                            (key, str(value)))
    return result


def param_type_check_float(key, value, type_):
    if isinstance(value, float):
        result = value
    else:
        try:
            result = float(value)
        except TypeError:
            raise TypeError("parameter: '%s' is '%s' but must be a %s" %
                            (key, str(value), str(type_)))
    if type_ == ParamType.FLOAT_NN:
        if result < 0:
            raise TypeError(("parameter: '%s' is '%s' "
                             + "but must be non-negative") %
                            (key, str(value)))
    return result


def checksum_file(logger, filename):
    """ Read file, compute checksum, return it as a string. """
    import zlib
    start = time.time()
    chunk_size = 10 * 1024 * 1024
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
    MB = total / (1024 * 1024)
    duration = stop - start
    rate = MB / duration
    logger.info("checksummed: %0.3f MB in %.3f seconds (%.2f MB/s)." %
                (MB, duration, rate))
    return str(checksum)


def param_allowed(key, value, allowed):
    """
    Check that the value is in the list of allowed values
    If allowed is None, there is no check, simply success
    """
    if allowed is None:
        return
    if value not in allowed:
        raise ValueError(("hyperparameter '%s'='%s' is not in the "
                          + "list of allowed values: %s") %
                         (key, value, str(allowed)))


def ckpt_parser(parser):
    # global
    parser.add_argument("--ckpt_restart_mode", type=str,
                        default='auto',
                        choices=['off', 'auto', 'required'],
                        help="Mode to restart from a saved checkpoint file, "
                             + "choices are 'off', 'auto', 'required'")
    parser.add_argument("--ckpt_checksum", type=str2bool,
                        default=False,
                        help="Checksum the restart file after read+write")
    parser.add_argument("--ckpt_skip_epochs", type=int,
                        default=0,
                        help="Number of epochs to skip before saving epochs")
    parser.add_argument("--ckpt_directory", type=str,
                        default='./save',
                        help="Base directory in which to save checkpoints")
    # saving
    parser.add_argument("--ckpt_save_best", type=str2bool,
                        default=True,
                        help="Toggle saving best model")
    parser.add_argument("--ckpt_save_best_metric", type=str,
                        default="val_loss",
                        help="Metric for determining when to save best model")
    parser.add_argument("--ckpt_save_weights_only", type=str2bool,
                        default=False,
                        help="Toggle saving only weights (not optimizer) (NYI)")
    parser.add_argument("--ckpt_save_interval", type=int,
                        default=1,
                        help="Interval to save checkpoints")
    # keeping
    parser.add_argument("--ckpt_keep_mode",
                        choices=['linear', 'exponential'],
                        help="Checkpoint saving mode. "
                             + "choices are 'linear' or 'exponential' ")
    parser.add_argument("--ckpt_keep_limit", type=int,
                        default=1000000,
                        help="Limit checkpoints to keep")

    return parser


def ckpt_defs(defs):
    # defs is an existing list
    # global
    new_defs = [
        {'name': 'ckpt_restart_mode',
            'type': str,
            'default': 'auto',
            'choices': ['off', 'auto', 'required'],
            'help': 'Mode to restart from a saved checkpoint file'},
        {'name': 'ckpt_checksum', 'type': str2bool,
            'default': False,
            'help': 'Checksum the restart file after read+write'},
        {'name': 'ckpt_skip_epochs', 'type': int,
            'default': 0,
            'help': 'Number of epochs to skip before saving epochs'},
        {'name': 'ckpt_directory', 'type': str,
            'default': './save',
            'help': 'Base directory in which to save checkpoints'},
        # saving
        {'name': 'ckpt_save_best',
            'type': str2bool,
            'default': True,
            'help': 'Toggle saving best model'},
        {'name': 'ckpt_save_best_metric',
            'type': str,
            'default': 'val_loss',
            'help': 'Metric for determining when to save best model'},
        {'name': 'ckpt_save_weights_only', 'type': str2bool,
            'default': False,
            'help': 'Toggle saving only weights (not optimizer) (NYI)'},
        {'name': 'ckpt_save_interval',
            'type': int,
            'default': 1,
            'help': 'Interval to save checkpoints'},
        # keeping
        {'name': 'ckpt_keep_mode',
            'choices': ['linear', 'exponential'],
            'help': 'Checkpoint saving mode. '
            + "choices are 'linear' or 'exponential' "},
        {'name': 'ckpt_keep_limit',
            'type': int,
            'default': 1000000,
            'help': 'Limit checkpoints to keep'}
    ]

    defs = defs + new_defs

    return defs
