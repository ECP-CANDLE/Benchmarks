#! /usr/bin/env python

"""Multilayer Perceptron for drug response problem"""

from __future__ import division, print_function

import argparse
import csv
import logging
import sys

import numpy as np
import pandas as pd

from itertools import tee, islice

from keras import backend as K
from keras import metrics
from keras.models import Sequential
from keras.layers import Activation, BatchNormalization, Dense, Dropout, LocallyConnected1D, Conv1D, MaxPooling1D, Flatten
from keras.callbacks import Callback, ModelCheckpoint, ProgbarLogger

from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler

# For non-interactive plotting
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import p1b3
from p1b3 import logger

# Model and Training parameters

# Seed for random generation
SEED = 2016
# Size of batch for training
BATCH_SIZE = 100
# Number of training epochs
EPOCHS = 20
# Number of data generator workers
WORKERS = 1

# Percentage of dropout used in training
DROP = 0.1
# Activation function (options: 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear')
ACTIVATION = 'relu'
LOSS = 'mse'
OPTIMIZER = 'sgd'
# OPTIMIZER = 'adam'

# Type of feature scaling (options: 'maxabs': to [-1,1]
#                                   'minmax': to [0,1]
#                                   None    : standard normalization
SCALING = 'std'
# Features to (randomly) sample from cell lines or drug descriptors
# FEATURE_SUBSAMPLE = 500
FEATURE_SUBSAMPLE = 0

# Number of units in fully connected (dense) layers
D1 = 1000
D2 = 500
D3 = 100
D4 = 50
DENSE_LAYERS = [D1, D2, D3, D4]

# Number of units per convolution layer or locally connected layer
CONV_LAYERS = [0, 0, 0] # filters, filter_len, stride
POOL = 10

MIN_LOGCONC = -5.
MAX_LOGCONC = -4.

CATEGORY_CUTOFFS = [0.]

VAL_SPLIT = 0.2
TEST_CELL_SPLIT = 0.15


np.set_printoptions(threshold=np.nan)
np.random.seed(SEED)


def get_parser():
    parser = argparse.ArgumentParser(prog='p1b3_baseline',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="increase output verbosity")
    parser.add_argument("-a", "--activation",
                        default=ACTIVATION,
                        help="keras activation function to use in inner layers: relu, tanh, sigmoid...")
    parser.add_argument("-e", "--epochs", type=int,
                        default=EPOCHS,
                        help="number of training epochs")
    parser.add_argument('-l', '--log', dest='logfile',
                        default=None,
                        help="log file")
    parser.add_argument("-z", "--batch_size", type=int,
                        default=BATCH_SIZE,
                        help="batch size")
    parser.add_argument("--batch_normalization", action="store_true",
                        help="use batch normalization")
    parser.add_argument("--conv", nargs='+', type=int,
                        default=CONV_LAYERS,
                        help="integer array describing convolution layers: conv1_filters, conv1_filter_len, conv1_stride, conv2_filters, conv2_filter_len, conv2_stride ...")
    parser.add_argument("--dense", nargs='+', type=int,
                        default=DENSE_LAYERS,
                        help="number of units in fully connected layers in an integer array")
    parser.add_argument("--drop", type=float,
                        default=DROP,
                        help="ratio of dropout used in fully connected layers")
    parser.add_argument("--locally_connected", action="store_true",
                        default=False,
                        help="use locally connected layers instead of convolution layers")
    parser.add_argument("--optimizer",
                        default=OPTIMIZER,
                        help="keras optimizer to use: sgd, rmsprop, ...")
    parser.add_argument("--loss",
                        default=LOSS,
                        help="keras loss function to use: mse, ...")
    parser.add_argument("--pool", type=int,
                        default=POOL,
                        help="pooling layer length")
    parser.add_argument("--scaling",
                        default=SCALING,
                        choices=['minabs', 'minmax', 'std', 'none'],
                        help="type of feature scaling; 'minabs': to [-1,1]; 'minmax': to [0,1], 'std': standard unit normalization; 'none': no normalization")
    parser.add_argument("--cell_features", nargs='+',
                        default=['expression'],
                        choices=['expression', 'mirna', 'proteome', 'all', 'categorical'],
                        help="use one or more cell line feature sets: 'expression', 'mirna', 'proteome', 'all'; or use 'categorical' for one-hot encoding of cell lines")
    parser.add_argument("--drug_features", nargs='+',
                        default=['descriptors'],
                        choices=['descriptors', 'latent', 'all', 'noise'],
                        help="use dragon7 descriptors, latent representations from Aspuru-Guzik's SMILES autoencoder, or both, or random features; 'descriptors','latent', 'all', 'noise'")
    parser.add_argument("--feature_subsample", type=int,
                        default=FEATURE_SUBSAMPLE,
                        help="number of features to randomly sample from each category (cellline expression, drug descriptors, etc), 0 means using all features")
    parser.add_argument("--min_logconc", type=float,
                        default=MIN_LOGCONC,
                        help="min log concentration of dose response data to use: -3.0 to -7.0")
    parser.add_argument("--max_logconc",  type=float,
                        default=MAX_LOGCONC,
                        help="max log concentration of dose response data to use: -3.0 to -7.0")
    parser.add_argument("--subsample",
                        default='naive_balancing',
                        choices=['naive_balancing', 'none'],
                        help="dose response subsample strategy; 'none' or 'naive_balancing'")
    parser.add_argument("--category_cutoffs", nargs='+', type=float,
                        default=CATEGORY_CUTOFFS,
                        help="list of growth cutoffs (between -1 and +1) seperating non-response and response categories")
    parser.add_argument("--val_split", type=float,
                        default=VAL_SPLIT,
                        help="fraction of data to use in validation")
    parser.add_argument("--test_cell_split", type=float,
                        default=TEST_CELL_SPLIT,
                        help="cell lines to use in test; if None use predefined unseen cell lines instead of sampling cell lines used in training")
    parser.add_argument("--train_steps", type=int,
                        default=0,
                        help="overrides the number of training batches per epoch if set to nonzero")
    parser.add_argument("--val_steps", type=int,
                        default=0,
                        help="overrides the number of validation batches per epoch if set to nonzero")
    parser.add_argument("--test_steps", type=int,
                        default=0,
                        help="overrides the number of test batches per epoch if set to nonzero")
    parser.add_argument("--save",
                        default='save',
                        help="prefix of output files")
    parser.add_argument("--scramble", action="store_true",
                        help="randomly shuffle dose response data")
    parser.add_argument("--workers", type=int,
                        default=WORKERS,
                        help="number of data generator workers")

    return parser


def extension_from_parameters(args):
    """Construct string for saving model with annotation of parameters"""
    ext = ''
    ext += '.A={}'.format(args.activation)
    ext += '.B={}'.format(args.batch_size)
    ext += '.D={}'.format(args.drop)
    ext += '.E={}'.format(args.epochs)
    if args.feature_subsample:
        ext += '.F={}'.format(args.feature_subsample)
    if args.conv:
        name = 'LC' if args.locally_connected else 'C'
        layer_list = list(range(0, len(args.conv), 3))
        for l, i in enumerate(layer_list):
            filters = args.conv[i]
            filter_len = args.conv[i+1]
            stride = args.conv[i+2]
            if filters <= 0 or filter_len <= 0 or stride <= 0:
                break
            ext += '.{}{}={},{},{}'.format(name, l+1, filters, filter_len, stride)
        if args.pool and args.conv[0] and args.conv[1]:
            ext += '.P={}'.format(args.pool)
    for i, n in enumerate(args.dense):
        if n:
            ext += '.D{}={}'.format(i+1, n)
    if args.batch_normalization:
        ext += '.BN'
    ext += '.S={}'.format(args.scaling)

    return ext


def evaluate_keras_metric(y_true, y_pred, metric):
    objective_function = metrics.get(metric)
    objective = objective_function(y_true, y_pred)
    return K.eval(objective)


def evaluate_model(model, generator, steps, metric, category_cutoffs=[0.]):
    y_true, y_pred = None, None
    count = 0
    while count < steps:
        x_batch, y_batch = next(generator)
        y_batch_pred = model.predict_on_batch(x_batch)
        y_batch_pred = y_batch_pred.ravel()
        y_true = np.concatenate((y_true, y_batch)) if y_true is not None else y_batch
        y_pred = np.concatenate((y_pred, y_batch_pred)) if y_pred is not None else y_batch_pred
        count += 1

    loss = evaluate_keras_metric(y_true.astype(np.float32), y_pred.astype(np.float32), metric)

    y_true_class = np.digitize(y_true, category_cutoffs)
    y_pred_class = np.digitize(y_pred, category_cutoffs)

    # theano does not like integer input
    acc = evaluate_keras_metric(y_true_class.astype(np.float32), y_pred_class.astype(np.float32), 'binary_accuracy')  # works for multiclass labels as well

    return loss, acc, y_true, y_pred, y_true_class, y_pred_class


def plot_error(y_true, y_pred, batch, file_ext, file_pre='save', subsample=1000):
    if batch % 10:
        return

    total = len(y_true)
    if subsample and subsample < total:
        usecols = np.random.choice(total, size=subsample, replace=False)
        y_true = y_true[usecols]
        y_pred = y_pred[usecols]

    y_true = y_true * 100
    y_pred = y_pred * 100
    diffs = y_pred - y_true

    bins = np.linspace(-200, 200, 100)
    if batch == 0:
        y_shuf = np.random.permutation(y_true)
        plt.hist(y_shuf - y_true, bins, alpha=0.5, label='Random')

    #plt.hist(diffs, bins, alpha=0.35-batch/100., label='Epoch {}'.format(batch+1))
    plt.hist(diffs, bins, alpha=0.3, label='Epoch {}'.format(batch+1))
    plt.title("Histogram of errors in percentage growth")
    plt.legend(loc='upper right')
    plt.savefig(file_pre+'.histogram'+file_ext+'.b'+str(batch)+'.png')
    plt.close()

    # Plot measured vs. predicted values
    fig, ax = plt.subplots()
    plt.grid('on')
    ax.scatter(y_true, y_pred, color='red', s=10)
    ax.plot([y_true.min(), y_true.max()],
            [y_true.min(), y_true.max()], 'k--', lw=4)
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    plt.savefig(file_pre+'.diff'+file_ext+'.b'+str(batch)+'.png')
    plt.close()


class MyLossHistory(Callback):
    def __init__(self, progbar, val_gen, test_gen, val_steps, test_steps, metric, category_cutoffs=[0.], ext='', pre='save'):
        super(MyLossHistory, self).__init__()
        self.progbar = progbar
        self.val_gen = val_gen
        self.test_gen = test_gen
        self.val_steps = val_steps
        self.test_steps = test_steps
        self.metric = metric
        self.category_cutoffs = category_cutoffs
        self.pre = pre
        self.ext = ext

    def on_train_begin(self, logs={}):
        self.best_val_loss = np.Inf
        self.best_val_acc = -np.Inf

    def on_epoch_end(self, batch, logs={}):
        val_loss, val_acc, y_true, y_pred, y_true_class, y_pred_class = evaluate_model(self.model, self.val_gen, self.val_steps, self.metric, self.category_cutoffs)
        test_loss, test_acc, _, _, _, _ = evaluate_model(self.model, self.test_gen, self.test_steps, self.metric, self.category_cutoffs)
        self.progbar.append_extra_log_values([('val_acc', val_acc), ('test_loss', test_loss), ('test_acc', test_acc)])
        if float(logs.get('val_loss', 0)) < self.best_val_loss:
            plot_error(y_true, y_pred, batch, self.ext, self.pre)
        self.best_val_loss = min(float(logs.get('val_loss', 0)), self.best_val_loss)
        self.best_val_acc = max(float(logs.get('val_acc', 0)), self.best_val_acc)


class MyProgbarLogger(ProgbarLogger):
    def __init__(self, samples):
        super(MyProgbarLogger, self).__init__(count_mode='samples')
        self.samples = samples

    def on_train_begin(self, logs=None):
        super(MyProgbarLogger, self).on_train_begin(logs)
        self.verbose = 1
        self.extra_log_values = []
        self.params['samples'] = self.samples

    def on_batch_begin(self, batch, logs=None):
        if self.seen < self.target:
            self.log_values = []
            self.extra_log_values = []

    def append_extra_log_values(self, tuples):
        for k, v in tuples:
            self.extra_log_values.append((k, v))

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        epoch_log = 'Epoch {}/{}'.format(epoch + 1, self.epochs)
        for k in self.params['metrics']:
            if k in logs:
                self.log_values.append((k, logs[k]))
                epoch_log += ' - {}: {:.4f}'.format(k, logs[k])
        for k, v in self.extra_log_values:
            self.log_values.append((k, v))
            epoch_log += ' - {}: {:.4f}'.format(k, v)
        if self.verbose:
            self.progbar.update(self.seen, self.log_values, force=True)
        logger.debug(epoch_log)


def main():
    parser = get_parser()
    args = parser.parse_args()

    ext = extension_from_parameters(args)

    logfile = args.logfile if args.logfile else args.save+ext+'.log'

    fh = logging.FileHandler(logfile)
    fh.setFormatter(logging.Formatter("[%(asctime)s %(process)d] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    fh.setLevel(logging.DEBUG)

    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter(''))
    sh.setLevel(logging.DEBUG if args.verbose else logging.INFO)

    logger.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    logger.addHandler(sh)

    logger.info('Args: {}'.format(args))

    loader = p1b3.DataLoader(val_split=args.val_split,
                             test_cell_split=args.test_cell_split,
                             cell_features=args.cell_features,
                             drug_features=args.drug_features,
                             feature_subsample=args.feature_subsample,
                             scaling=args.scaling,
                             scramble=args.scramble,
                             min_logconc=args.min_logconc,
                             max_logconc=args.max_logconc,
                             subsample=args.subsample,
                             category_cutoffs=args.category_cutoffs)

    gen_shape = None
    out_dim = 1

    model = Sequential()
    if args.conv and args.conv[0]:
        gen_shape = 'add_1d'
        layer_list = list(range(0, len(args.conv), 3))
        for l, i in enumerate(layer_list):
            filters = args.conv[i]
            filter_len = args.conv[i+1]
            stride = args.conv[i+2]
            if filters <= 0 or filter_len <= 0 or stride <= 0:
                break
            if args.locally_connected:
                model.add(LocallyConnected1D(filters, filter_len, strides=stride, input_shape=(loader.input_dim, 1)))
            else:
                model.add(Conv1D(filters, filter_len, strides=stride, input_shape=(loader.input_dim, 1)))
            if args.batch_normalization:
                model.add(BatchNormalization())
            model.add(Activation(args.activation))
            if args.pool:
                model.add(MaxPooling1D(pool_size=args.pool))
        model.add(Flatten())

    for layer in args.dense:
        if layer:
            model.add(Dense(layer, input_dim=loader.input_dim))
            if args.batch_normalization:
                model.add(BatchNormalization())
            model.add(Activation(args.activation))
            if args.drop:
                model.add(Dropout(args.drop))
    model.add(Dense(out_dim))

    model.summary()
    logger.debug('Model: {}'.format(model.to_json()))

    model.compile(loss=args.loss, optimizer=args.optimizer)

    train_gen = p1b3.DataGenerator(loader, batch_size=args.batch_size, shape=gen_shape, name='train_gen').flow()
    val_gen = p1b3.DataGenerator(loader, partition='val', batch_size=args.batch_size, shape=gen_shape, name='val_gen').flow()
    val_gen2 = p1b3.DataGenerator(loader, partition='val', batch_size=args.batch_size, shape=gen_shape, name='val_gen2').flow()
    test_gen = p1b3.DataGenerator(loader, partition='test', batch_size=args.batch_size, shape=gen_shape, name='test_gen').flow()

    train_steps = int(loader.n_train/args.batch_size)
    val_steps = int(loader.n_val/args.batch_size)
    test_steps = int(loader.n_test/args.batch_size)

    train_steps = args.train_steps if args.train_steps else train_steps
    val_steps = args.val_steps if args.val_steps else val_steps
    test_steps = args.test_steps if args.test_steps else test_steps

    checkpointer = ModelCheckpoint(filepath=args.save+'.model'+ext+'.h5', save_best_only=True)
    progbar = MyProgbarLogger(train_steps * args.batch_size)
    history = MyLossHistory(progbar=progbar, val_gen=val_gen2, test_gen=test_gen,
                            val_steps=val_steps, test_steps=test_steps,
                            metric=args.loss, category_cutoffs=args.category_cutoffs,
                            ext=ext, pre=args.save)

    model.fit_generator(train_gen, train_steps,
                        epochs=args.epochs,
                        validation_data=val_gen,
                        validation_steps=val_steps,
                        verbose=0,
                        callbacks=[checkpointer, history, progbar],
                        pickle_safe=True,
                        workers=args.workers)


if __name__ == '__main__':
    main()
    try:
        K.clear_session()
    except AttributeError:      # theano does not have this function
        pass
