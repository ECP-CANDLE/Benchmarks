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
from keras.layers import Dense, Dropout, LocallyConnected1D, Convolution1D, MaxPooling1D, Flatten
from keras.callbacks import Callback, ModelCheckpoint, ProgbarLogger

from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler

# For non-interactive plotting
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import p1b3


# Model and Training parameters

# Seed for random generation
SEED = 2016
# Size of batch for training
BATCH_SIZE = 100
# Number of training epochs
NB_EPOCH = 20
# Number of data generator workers
NB_WORKER = 1

# Percentage of dropout used in training
DROP = 0.1
# Activation function (options: 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear')
ACTIVATION = 'relu'
LOSS = 'mse'
OPTIMIZER = 'adam'

# Type of feature scaling (options: 'maxabs': to [-1,1]
#                                   'minmax': to [0,1]
#                                   None    : standard normalization
SCALING = 'std'
# Features to (randomly) sample from cell lines or drug descriptors
FEATURE_SUBSAMPLE = 500#0
# FEATURE_SUBSAMPLE = 0

# Number of units in fully connected (dense) layers
D1 = 1000
D2 = 500
D3 = 100
D4 = 50
DENSE_LAYERS = [D1, D2, D3, D4]

# Number of units per locally connected layer
LC1 = 10, 10        # nb_filter, filter_length
LC2 = 0, 0         # disabled layer
# LOCALLY_CONNECTED_LAYERS = list(LC1 + LC2)
LOCALLY_CONNECTED_LAYERS = [0, 0]
POOL = 100

MIN_LOGCONC = -5.
MAX_LOGCONC = -4.

CATEGORY_CUTOFFS = [0.]

np.set_printoptions(threshold=np.nan)
np.random.seed(SEED)


def get_parser():
    parser = argparse.ArgumentParser(prog='p1b3_baseline',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="increase output verbosity")
    parser.add_argument("-a", "--activation", action="store",
                        default=ACTIVATION,
                        help="keras activation function to use in inner layers: relu, tanh, sigmoid...")
    parser.add_argument("-b", "--batch_size", action="store",
                        default=BATCH_SIZE, type=int,
                        help="batch size")
    parser.add_argument("-c", "--convolution", action="store_true",
                        default=False,
                        help="use convolution layers instead of locally connected layers")
    parser.add_argument("-d", "--dense", action="store", nargs='+', type=int,
                        default=DENSE_LAYERS,
                        help="number of units in fully connected layers in an integer array")
    parser.add_argument("-e", "--epochs", action="store",
                        default=NB_EPOCH, type=int,
                        help="number of training epochs")
    parser.add_argument("-l", "--locally_connected", action="store", nargs='+', type=int,
                        default=LOCALLY_CONNECTED_LAYERS,
                        help="integer array describing locally connected layers: layer1_nb_filter, layer1_filter_len, layer2_nb_filter, layer2_filter_len, ...")
    parser.add_argument("-o", "--optimizer", action="store",
                        default=OPTIMIZER,
                        help="keras optimizer to use: sgd, rmsprop, ...")
    parser.add_argument("--drop", action="store",
                        default=DROP, type=float,
                        help="ratio of dropout used in fully connected layers")
    parser.add_argument("--loss", action="store",
                        default=LOSS,
                        help="keras loss function to use: mse, ...")
    parser.add_argument("--pool", action="store",
                        default=POOL, type=int,
                        help="pooling layer length")
    parser.add_argument("--scaling", action="store",
                        default=SCALING,
                        help="type of feature scaling; 'minabs': to [-1,1]; 'minmax': to [0,1], 'std': standard unit normalization; None: no normalization")
    parser.add_argument("--drug_features", action="store",
                        default="descriptors",
                        help="use dragon7 descriptors, latent representations from Aspuru-Guzik's SMILES autoencoder, or both, or random features; 'descriptors','latent', 'both', 'noise'")
    parser.add_argument("--feature_subsample", action="store",
                        default=FEATURE_SUBSAMPLE, type=int,
                        help="number of features to randomly sample from each category (cellline expression, drug descriptors, etc), 0 means using all features")
    parser.add_argument("--min_logconc", action="store",
                        default=MIN_LOGCONC, type=float,
                        help="min log concentration of dose response data to use: -3.0 to -7.0")
    parser.add_argument("--max_logconc", action="store",
                        default=MAX_LOGCONC, type=float,
                        help="max log concentration of dose response data to use: -3.0 to -7.0")
    parser.add_argument("--subsample", action="store",
                        default='naive_balancing',
                        help="dose response subsample strategy; None or 'naive_balancing'")
    parser.add_argument("--category_cutoffs", action="store", nargs='+', type=float,
                        default=CATEGORY_CUTOFFS,
                        help="list of growth cutoffs (between -1 and +1) seperating non-response and response categories")
    parser.add_argument("--train_samples", action="store",
                        default=0, type=int,
                        help="overrides the number of training samples if set to nonzero")
    parser.add_argument("--val_samples", action="store",
                        default=0, type=int,
                        help="overrides the number of validation samples if set to nonzero")
    parser.add_argument("--save", action="store",
                        default='save',
                        help="prefix of output files")
    parser.add_argument("--scramble", action="store_true",
                        help="randomly shuffle dose response data")
    parser.add_argument("--workers", action="store",
                        default=NB_WORKER, type=int,
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
    if args.locally_connected:
        name = 'C' if args.convolution else 'LC'
        layer_list = list(range(0, len(args.locally_connected), 2))
        for l, i in enumerate(layer_list):
            nb_filter = args.locally_connected[i]
            filter_len = args.locally_connected[i+1]
            if nb_filter <= 0 or filter_len <= 0:
                break
            ext += '.{}{}={},{}'.format(name, l+1, nb_filter, filter_len)
        if args.pool and layer_list[0] and layer_list[1]:
            ext += '.P={}'.format(args.pool)
    for i, n in enumerate(args.dense):
        if n:
            ext += '.D{}={}'.format(i+1, n)
    ext += '.S={}'.format(args.scaling)

    return ext


def evaluate_keras_metric(y_true, y_pred, metric):
    objective_function = metrics.get(metric)
    objective = objective_function(y_true, y_pred)
    return K.eval(objective)


def evaluate_model(model, generator, samples, metric, category_cutoffs=[0.]):
    y_true, y_pred = None, None
    count = 0
    while count < samples:
        x_batch, y_batch = next(generator)
        y_batch_pred = model.predict_on_batch(x_batch)
        y_batch_pred = y_batch_pred.ravel()
        y_true = np.concatenate((y_true, y_batch)) if y_true is not None else y_batch
        y_pred = np.concatenate((y_pred, y_batch_pred)) if y_pred is not None else y_batch_pred
        count += len(y_batch)

    loss = evaluate_keras_metric(y_true, y_pred, metric)

    y_true_class = np.digitize(y_true, category_cutoffs)
    y_pred_class = np.digitize(y_pred, category_cutoffs)

    acc = evaluate_keras_metric(y_true_class, y_pred_class, 'binary_accuracy')  # works for multiclass labels as well

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
    def __init__(self, progbar, val_gen, test_gen, val_samples, test_samples, metric, category_cutoffs=[0.], ext='', pre='save'):
        super(MyLossHistory, self).__init__()
        self.progbar = progbar
        self.val_gen = val_gen
        self.test_gen = test_gen
        self.val_samples = val_samples
        self.test_samples = test_samples
        self.metric = metric
        self.category_cutoffs = category_cutoffs
        self.pre = pre
        self.ext = ext

    def on_train_begin(self, logs={}):
        self.best_val_loss = np.Inf
        self.best_val_acc = -np.Inf
        self.best_model = None

    def on_epoch_end(self, batch, logs={}):
        if float(logs.get('val_loss', 0)) < self.best_val_loss:
            self.best_model = self.model
            val_loss, val_acc, y_true, y_pred, y_true_class, y_pred_class = evaluate_model(self.best_model, self.val_gen, self.val_samples, self.metric, self.category_cutoffs)
            test_loss, test_acc, _, _, _, _ = evaluate_model(self.best_model, self.test_gen, self.test_samples, self.metric, self.category_cutoffs)
            self.progbar.append_extra_log_values([('val_acc', val_acc), ('test_loss', test_loss), ('test_acc', test_acc)])
            plot_error(y_true, y_pred, batch, self.ext, self.pre)
        self.best_val_loss = min(float(logs.get('val_loss', 0)), self.best_val_loss)
        self.best_val_acc = max(float(logs.get('val_acc', 0)), self.best_val_acc)


class MyProgbarLogger(ProgbarLogger):
    def on_train_begin(self, logs=None):
        super(MyProgbarLogger, self).on_train_begin(logs)
        self.verbose = 1
        self.extra_log_values = []

    def append_extra_log_values(self, tuples):
        for k, v in tuples:
            self.extra_log_values.append((k, v))

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        for k in self.params['metrics']:
            if k in logs:
                self.log_values.append((k, logs[k]))
        for k, v in self.extra_log_values:
            self.log_values.append((k, v))
        if self.verbose:
            self.progbar.update(self.seen, self.log_values, force=True)


def main():
    parser = get_parser()
    args = parser.parse_args()
    print('Args:', args)

    loggingLevel = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=loggingLevel, format='')

    ext = extension_from_parameters(args)

    datagen = p1b3.RegressionDataGenerator(feature_subsample=args.feature_subsample,
                                           scaling=args.scaling,
                                           drug_features=args.drug_features,
                                           scramble=args.scramble,
                                           min_logconc=args.min_logconc,
                                           max_logconc=args.max_logconc,
                                           subsample=args.subsample,
                                           category_cutoffs=args.category_cutoffs)

    topology = 'dense'
    out_dim = 1

    model = Sequential()
    if args.locally_connected and args.locally_connected[0]:
        topology = 'simple_local'
        layer_list = list(range(0, len(args.locally_connected), 2))
        for l, i in enumerate(layer_list):
            nb_filter = args.locally_connected[i]
            filter_len = args.locally_connected[i+1]
            if nb_filter <= 0 or filter_len <= 0:
                break
            if args.convolution:
                model.add(Convolution1D(nb_filter, filter_len, input_shape=(datagen.input_dim, 1), activation=args.activation))
            else:
                model.add(LocallyConnected1D(nb_filter, filter_len, input_shape=(datagen.input_dim, 1), activation=args.activation))
            if args.pool:
                model.add(MaxPooling1D(pool_length=args.pool))
        model.add(Flatten())

    for layer in args.dense:
        if layer:
            model.add(Dense(layer, input_dim=datagen.input_dim, activation=args.activation))
            if args.drop:
                model.add(Dropout(args.drop))
    model.add(Dense(out_dim))

    model.summary()
    model.compile(loss=args.loss, optimizer=args.optimizer)

    train_gen = datagen.flow(batch_size=args.batch_size, topology=topology)
    val_gen = datagen.flow(data='val', batch_size=args.batch_size, topology=topology)
    val_gen2 = datagen.flow(data='val', batch_size=args.batch_size, topology=topology)
    test_gen = datagen.flow(data='test', batch_size=args.batch_size, topology=topology)

    train_samples = int(datagen.n_train/args.batch_size) * args.batch_size
    val_samples = int(datagen.n_val/args.batch_size) * args.batch_size
    test_samples = int(datagen.n_test/args.batch_size) * args.batch_size

    train_samples = args.train_samples if args.train_samples else train_samples
    val_samples = args.val_samples if args.val_samples else val_samples

    checkpointer = ModelCheckpoint(filepath=args.save+'.model'+ext+'.h5', save_best_only=True)
    progbar = MyProgbarLogger()
    history = MyLossHistory(progbar=progbar, val_gen=val_gen2, test_gen=test_gen,
                            val_samples=val_samples, test_samples=test_samples,
                            metric=args.loss, category_cutoffs=args.category_cutoffs,
                            ext=ext, pre=args.save)

    model.fit_generator(train_gen, train_samples,
                        nb_epoch=args.epochs,
                        validation_data=val_gen,
                        nb_val_samples=val_samples,
                        verbose=0,
                        callbacks=[checkpointer, history, progbar],
                        pickle_safe=True,
                        nb_worker=args.workers)


if __name__ == '__main__':
    main()
    K.clear_session()
