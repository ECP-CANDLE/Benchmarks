from __future__ import print_function

import argparse
import logging
import os

import numpy as np
from sklearn.metrics import accuracy_score

file_path = os.path.dirname(os.path.realpath(__file__))

import candle

logger = logging.getLogger(__name__)

additional_definitions = [
    {
        "name": "reg_l2",
        "type": float,
        "default": 0.0,
        "help": "weight of regularization for l2 norm of nn weights",
    },
    {
        "name": "one_hot_dtrep",
        "type": candle.str2bool,
        "default": True,
        "help": "use one-hot data representation.",
    }
]

temporary = [
    {
        "name": "saved_model",
        "type": str,
        "default": argparse.SUPPRESS,
        "help": "file with saved model.",
    },
    {
        "name": "saved_weights",
        "type": str,
        "default": argparse.SUPPRESS,
        "help": "file with model's saved weights.",
    },
]

required = [
    "data_url",
    "train_data",
    "test_data",
    "activation",
    "batch_size",
    "dense",
    "dropout",
    "epochs",
    "feature_subsample",
    "initialization",
    "learning_rate",
    "loss",
    "optimizer",
    "reg_l2",
    "rng_seed",
    "scaling",
    "val_split",
    "shuffle",
]


class BenchmarkP1B2(candle.Benchmark):
    def set_locals(self):
        """Functionality to set variables specific for the benchmark
        - required: set of required parameters for the benchmark.
        - additional_definitions: list of dictionaries describing the additional parameters for the
        benchmark.
        """

        if required is not None:
            self.required = set(required)
        if additional_definitions is not None:
            self.additional_definitions = additional_definitions + temporary


def extension_from_parameters(params, framework):
    """Construct string for saving model with annotation of parameters"""
    ext = framework
    ext += ".A={}".format(params["activation"])
    ext += ".B={}".format(params["batch_size"])
    ext += ".D={}".format(params["dropout"])
    ext += ".E={}".format(params["epochs"])
    if params["feature_subsample"]:
        ext += ".F={}".format(params["feature_subsample"])
    for i, n in enumerate(params["dense"]):
        if n:
            ext += ".D{}={}".format(i + 1, n)
    ext += ".S={}".format(params["scaling"])

    return ext


def load_data(params):

    # fetch data
    file_train = candle.fetch_file(
        params["data_url"] + params["train_data"], subdir="Pilot1"
    )
    file_test = candle.fetch_file(
        params["data_url"] + params["test_data"], subdir="Pilot1"
    )

    seed = params["rng_seed"]
    one_hot_dtrep = params["one_hot_dtrep"]

    if one_hot_dtrep:  # Use one-hot data representation
        return candle.load_Xy_one_hot_data2(
            file_train,
            file_test,
            class_col=["cancer_type"],
            drop_cols=["case_id", "cancer_type"],
            n_cols=params["feature_subsample"],
            shuffle=params["shuffle"],
            scaling=params["scaling"],
            validation_split=params["val_split"],
            dtype=params["data_type"],
            seed=seed,
        )

    return candle.load_Xy_data2(
        file_train,
        file_test,
        class_col=["cancer_type"],
        drop_cols=["case_id", "cancer_type"],
        n_cols=params["feature_subsample"],
        shuffle=params["shuffle"],
        scaling=params["scaling"],
        validation_split=params["val_split"],
        dtype=params["data_type"],
        seed=seed,
    )


def evaluate_accuracy(y_pred, y_test, one_hot_dtrep=False):
    if one_hot_dtrep:
        def map_max_indices(nparray):
            # maxi = lambda a: a.argmax()
            def maxi(a):
                return a.argmax()

            # iter_to_na = lambda i: np.fromiter(i, dtype=np.float)
            return np.array([maxi(a) for a in nparray])

        ya, ypa = tuple(map(map_max_indices, (y_test, y_pred)))
        accuracy = accuracy_score(ya, ypa)
    else:
        accuracy = accuracy_score(y_test, y_pred)
    # print('Accuracy: {}%'.format(100 * accuracy))
    return {"accuracy": accuracy}
