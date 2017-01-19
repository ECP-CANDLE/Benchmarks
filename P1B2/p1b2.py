from __future__ import absolute_import
from data_utils import get_file
import numpy as np
import os
import pandas as pd

import gzip
# from ..utils.data_utils import get_file
# from six.moves import cPickle
import sys

from sklearn.metrics import accuracy_score


seed = 2016


def load_data(shuffle=True, n_cols=None):
    train_path = get_file('P1B2.train.csv', origin='http://ftp.mcs.anl.gov/pub/candle/public/benchmarks/P1B2/P1B2.train.csv')
    test_path = get_file('P1B2.test.csv', origin='http://ftp.mcs.anl.gov/pub/candle/public/benchmarks/P1B2/P1B2.test.csv')

    usecols = list(range(n_cols)) if n_cols else None

    df_train = pd.read_csv(train_path, engine='c', usecols=usecols)
    df_test = pd.read_csv(test_path, engine='c', usecols=usecols)

    if shuffle:
        df_train = df_train.sample(frac=1, random_state=seed)
        df_test = df_test.sample(frac=1, random_state=seed)

    X_train = df_train.iloc[:, 2:].as_matrix()
    X_test = df_test.iloc[:, 2:].as_matrix()

    y_train = pd.get_dummies(df_train[['cancer_type']]).as_matrix()
    y_test = pd.get_dummies(df_test[['cancer_type']]).as_matrix()

    return (X_train, y_train), (X_test, y_test)


def evaluate(y_pred, y_test):
    def map_max_indices(nparray):
        maxi = lambda a: a.argmax()
        iter_to_na = lambda i: np.fromiter(i, dtype=np.float)
        return np.array([maxi(a) for a in nparray])
    ya, ypa = tuple(map(map_max_indices, (y_test, y_pred)))
    accuracy = accuracy_score(ya, ypa)
    # print('Final accuracy of best model: {}%'.format(100 * accuracy))
    return {'accuracy': accuracy}
