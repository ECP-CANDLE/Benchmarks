import typing
from collections.abc import abc

import torch
import numpy as np
import pandas as pd


class TrainingHistory:

    def __init__(self):
        self.data = []

    def add(self, epoch_result):
        """ Add a datapoint to the history """
        self.data.append(epoch_result)

    def frame(self):
        return pd.DataFrame(self.data).set_index('epoch_index')


class TrainingInfo(abc.MutableMapping):
    """ Information that needs to persist through training """

    def __init__(self, start_epoch_index=0, run_name: typing.Optional[str] = None, metrics=None, callbacks=None):
        self.data_dict = {}  # optional information

        self.run_name = run_name
        self.history = TrainingHistory()
        self.start_epoch_index = start_epoch_index
        self.metrics = metrics if metrics is not None else []
        self.callbacks = callbacks if callbacks is not None else []

    def initialize(self):
        for callback in self.callbacks:
            callback.on_initialization(self)

    def on_train_begin(self):
        """ Start the training process - always used, even in restarts """
        for callback in self.callbacks:
            callback.on_train_begin(self)

    def on_train_end(self):
        """ Finalize training process """
        for callback in self.callbacks:
            callback.on_train_end(self)

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value

    def __delitem__(self, key):
        del self.data[key]

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)

    def __contains__(self, key):
        return key in self.data


class EpochResultAccumulator(abc.MutableMapping):
    """ Result of a single epoch of training """

    def __init__(self, global_epoch_idx, metrics):
        self.metrics = metrics
        self.global_epoch_idx = global_epoch_idx
