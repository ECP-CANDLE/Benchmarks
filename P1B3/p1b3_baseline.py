#! /usr/bin/env python

"""Multilayer Perceptron for drug response problem"""

from __future__ import division, print_function

import csv
import logging

import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import Callback, ModelCheckpoint

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

# Percentage of drop used in training
DROP = 0.1
# Activation function (options: 'relu', 'tanh', 'sigmoid', 'hard_sigmoid',
#                               'linear')
ACTIVATION = 'relu'

# Type of feature scaling (options: 'maxabs': to [-1,1]
#                                   'minmax': to [0,1]
#                                   None    : standard normalization
SCALING = 'minmax'
# Features to (randomly) sample from cell lines or drug descriptors
FEATURE_SUBSAMPLE = 500#0

# Number of units per layer
L1 = 1000
L2 = 500
L3 = 100
L4 = 50
LAYERS = [L1, L2, L3, L4]


np.set_printoptions(threshold=np.nan)
np.random.seed(SEED)


def extension_from_parameters():
    """Construct string for saving model with annotation of parameters"""
    ext = ''
    ext += '.A={}'.format(ACTIVATION)
    ext += '.B={}'.format(BATCH_SIZE)
    ext += '.D={}'.format(DROP)
    ext += '.E={}'.format(NB_EPOCH)
    if FEATURE_SUBSAMPLE:
        ext += '.F={}'.format(FEATURE_SUBSAMPLE)
    for i, n in enumerate(LAYERS):
        if n:
            ext += '.L{}={}'.format(i+1, n)

    ext += '.S={}'.format(SCALING)

    return ext


def plot_error(model, generator, samples, batch, file_ext):
    if batch % 20:
        return
    fig2 = plt.figure()
    samples = 100
    diffs = None
    y_all = None
    for i in range(samples):
        X, y = next(generator)
        y = y.ravel() * 100
        y_pred = model.predict(X)
        y_pred = y_pred.ravel() * 100
        diff =  y - y_pred
        diffs = np.concatenate((diffs, diff)) if diffs is not None else diff
        y_all = np.concatenate((y_all, y)) if y_all is not None else y
    bins = np.linspace(-200, 200, 100)
    if batch == 0:
        y_shuf = np.random.permutation(y_all)
        plt.hist(y_all - y_shuf, bins, alpha=0.5, label='Random')
    #plt.hist(diffs, bins, alpha=0.35-batch/100., label='Epoch {}'.format(batch+1))
    plt.hist(diffs, bins, alpha=0.3, label='Epoch {}'.format(batch+1))
    plt.title("Histogram of errors in percentage growth")
    plt.legend(loc='upper right')
    plt.savefig('plot'+file_ext+'.b'+str(batch)+'.png')
    # plt.savefig('plot'+file_ext+'.png')
    plt.close()

    # Plot measured vs. predicted values
    fig, ax = plt.subplots()
    plt.grid('on')
    ax.scatter(y, y_pred, color='red')
    ax.plot([y.min(), y.max()],
            [y.min(), y.max()], 'k--', lw=4)
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    #plt.show()
    plt.savefig('meas_vs_pred'+file_ext+'.b'+str(batch)+'.png')
    plt.close()


class BestLossHistory(Callback):
    def __init__(self, generator, samples, ext):
        super(BestLossHistory, self).__init__()
        self.generator = generator
        self.samples = samples
        self.ext = ext

    def on_train_begin(self, logs={}):
        self.best_val_loss = np.Inf
        self.best_val_acc = -np.Inf
        self.best_model = None

    def on_epoch_end(self, batch, logs={}):
        if float(logs.get('val_loss', 0)) < self.best_val_loss:
            self.best_model = self.model
            plot_error(self.best_model, self.generator, self.samples, batch, self.ext)
        self.best_val_loss = min(float(logs.get('val_loss', 0)), self.best_val_loss)
        self.best_val_acc = max(float(logs.get('val_acc', 0)), self.best_val_acc)


def main():
    ext = extension_from_parameters()

    out_dim = 1
    loss = 'mse'
    metrics = None
    #metrics = ['accuracy'] if CATEGORICAL else None

    datagen = p1b3.RegressionDataGenerator(feature_subsample=FEATURE_SUBSAMPLE, scaling=SCALING)
    train_gen = datagen.flow(batch_size=BATCH_SIZE)
    val_gen = datagen.flow(val=True, batch_size=BATCH_SIZE)
    val_gen2 = datagen.flow(val=True, batch_size=BATCH_SIZE)

    model = Sequential()
    model.add(Dense(LAYERS[0], input_dim=datagen.input_dim, activation=ACTIVATION))
    for layer in LAYERS[1:]:
        if layer:
            if DROP:
                model.add(Dropout(DROP))
            model.add(Dense(layer, activation=ACTIVATION))
    model.add(Dense(out_dim, activation=ACTIVATION))

    model.summary()
    model.compile(loss=loss, optimizer='rmsprop', metrics=metrics)

    train_samples = int(datagen.n_train/BATCH_SIZE) * BATCH_SIZE
    val_samples = int(datagen.n_val/BATCH_SIZE) * BATCH_SIZE

    history = BestLossHistory(val_gen2, val_samples, ext)
    checkpointer = ModelCheckpoint(filepath='model'+ext+'.h5', save_best_only=True)

    model.fit_generator(train_gen, train_samples,
                        nb_epoch = NB_EPOCH,
                        validation_data = val_gen,
                        nb_val_samples = val_samples,
                        callbacks=[history, checkpointer])
                        # nb_worker = 1)


if __name__ == '__main__':
    main()
