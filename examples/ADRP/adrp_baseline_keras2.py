from __future__ import print_function

import itertools
import pandas as pd
import numpy as np
import os
import sys
import gzip
import argparse
import sklearn

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

import tensorflow as tf

import keras as ke
from keras import backend as K

from keras.layers import Input, Dense, Dropout, Activation, BatchNormalization
from keras.optimizers import SGD, Adam, RMSprop, Adadelta
from keras.models import Sequential, Model, model_from_json, model_from_yaml
from keras.utils import np_utils, multi_gpu_model

from keras.callbacks import (
    Callback,
    ModelCheckpoint,
    CSVLogger,
    ReduceLROnPlateau,
    EarlyStopping,
    TensorBoard,
)

from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    roc_auc_score,
    confusion_matrix,
    balanced_accuracy_score,
    classification_report,
)
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler
from sklearn.metrics import (
    recall_score,
    auc,
    roc_curve,
    f1_score,
    precision_recall_curve,
)

import adrp
import candle

np.set_printoptions(precision=4)


def r2(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - SS_res / (SS_tot + K.epsilon())


def tf_auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc


# from sklearn.metrics import roc_auc_score
# import tensorflow as tf


def auroc(y_true, y_pred):
    score = tf.py_func(
        lambda y_true, y_pred: roc_auc_score(
            y_true, y_pred, average="macro", sample_weight=None
        ).astype("float32"),
        [y_true, y_pred],
        "float32",
        stateful=False,
        name="sklearnAUC",
    )
    return score


#def covariance(x, y):
#    return K.mean(x * y) - K.mean(x) * K.mean(y)


def corr(y_true, y_pred):
    cov = candle.covariance(y_true, y_pred)
    var1 = candle.covariance(y_true, y_true)
    var2 = candle.covariance(y_pred, y_pred)
    return cov / (K.sqrt(var1 * var2) + K.epsilon())


#def xent(y_true, y_pred):
#    return binary_crossentropy(y_true, y_pred)


#def mse(y_true, y_pred):
#    return mean_squared_error(y_true, y_pred)


class MetricHistory(Callback):
    def on_epoch_begin(self, epoch, logs=None):
        print("\n")

    def on_epoch_end(self, epoch, logs=None):
        y_pred = self.model.predict(self.validation_data[0])
        r2 = r2_score(self.validation_data[1], y_pred)
        corr, _ = pearsonr(self.validation_data[1].flatten(), y_pred.flatten())
        print("\nval_r2:", r2)
        print(y_pred.shape)
        print("\nval_corr:", corr, "val_r2:", r2)
        print("\n")


class LoggingCallback(Callback):
    def __init__(self, print_fcn=print):
        Callback.__init__(self)
        self.print_fcn = print_fcn

    def on_epoch_end(self, epoch, logs={}):
        msg = "[Epoch: %i] %s" % (
            epoch,
            ", ".join("%s: %f" % (k, v) for k, v in sorted(logs.items())),
        )
        self.print_fcn(msg)


def build_type_classifier(x_train, y_train, x_test, y_test):
    y_train = np.argmax(y_train, axis=1)
    y_test = np.argmax(y_test, axis=1)
    from xgboost import XGBClassifier

    clf = XGBClassifier(max_depth=6, n_estimators=100)
    clf.fit(
        x_train, y_train, eval_set=[(x_train, y_train), (x_test, y_test)], verbose=False
    )
    y_pred = clf.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    print(acc)
    return clf


def initialize_parameters(default_model="adrp_default_model.txt"):

    # Build benchmark object
    adrpBmk = adrp.BenchmarkAdrp(
        adrp.file_path,
        default_model,
        "keras",
        prog="adrp_baseline",
        desc="Multi-task (DNN) for data extraction from clinical reports - Pilot 3 Benchmark 1",
    )

    # Initialize parameters
    gParameters = candle.finalize_parameters(adrpBmk)
    # adrp.logger.info('Params: {}'.format(gParameters))

    return gParameters


def save_cache(
    cache_file, x_train, y_train, x_val, y_val, x_test, y_test, x_labels, y_labels
):
    with h5py.File(cache_file, "w") as hf:
        hf.create_dataset("x_train", data=x_train)
        hf.create_dataset("y_train", data=y_train)
        hf.create_dataset("x_val", data=x_val)
        hf.create_dataset("y_val", data=y_val)
        hf.create_dataset("x_test", data=x_test)
        hf.create_dataset("y_test", data=y_test)
        hf.create_dataset(
            "x_labels",
            (len(x_labels), 1),
            "S100",
            data=[x.encode("ascii", "ignore") for x in x_labels],
        )
        hf.create_dataset(
            "y_labels",
            (len(y_labels), 1),
            "S100",
            data=[x.encode("ascii", "ignore") for x in y_labels],
        )


def load_cache(cache_file):
    with h5py.File(cache_file, "r") as hf:
        x_train = hf["x_train"][:]
        y_train = hf["y_train"][:]
        x_val = hf["x_val"][:]
        y_val = hf["y_val"][:]
        x_test = hf["x_test"][:]
        y_test = hf["y_test"][:]
        x_labels = [x[0].decode("unicode_escape") for x in hf["x_labels"][:]]
        y_labels = [x[0].decode("unicode_escape") for x in hf["y_labels"][:]]
    return x_train, y_train, x_val, y_val, x_test, y_test, x_labels, y_labels


def run(params):
    args = candle.ArgumentStruct(**params)
    seed = args.rng_seed
    candle.set_seed(seed)

    # Construct extension to save model
    ext = adrp.extension_from_parameters(params, ".keras")
    params['save_path'] = './'+params['base_name']+'/'
    candle.verify_path(params["save_path"])
    prefix = "{}{}".format(params["save_path"], ext)
    logfile = params["logfile"] if params["logfile"] else prefix + ".log"
    candle.set_up_logger(logfile, adrp.logger, params["verbose"])
    adrp.logger.info("Params: {}".format(params))

    # Get default parameters for initialization and optimizer functions
    keras_defaults = candle.keras_default_config()

    ##
    X_train, Y_train, X_test, Y_test, PS = adrp.load_data(params, seed)

    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)

    print("Y_train shape:", Y_train.shape)
    print("Y_test shape:", Y_test.shape)

    # Initialize weights and learning rule
    initializer_weights = candle.build_initializer(
        params["initialization"], keras_defaults, seed
    )
    initializer_bias = candle.build_initializer("constant", keras_defaults, 0.0)

    activation = params["activation"]
    out_activation = params["out_activation"]

    # TODO: set output_dim
    output_dim = 1

    # TODO: Use dense_layers for creating inputs/outputs
    dense_layers = params["dense"]

    inputs = Input(shape=(PS,))

    if dense_layers != None:
        if type(dense_layers) != list:
            dense_layers = list(dense_layers)
        for i, l in enumerate(dense_layers):
            if i == 0:
                x = Dense(
                    l,
                    activation=activation,
                    kernel_initializer=initializer_weights,
                    bias_initializer=initializer_bias,
                )(inputs)
            else:
                x = Dense(
                    l,
                    activation=activation,
                    kernel_initializer=initializer_weights,
                    bias_initializer=initializer_bias,
                )(x)
            if params["dropout"]:
                x = Dropout(params["dropout"])(x)
        output = Dense(
            output_dim,
            activation=out_activation,
            kernel_initializer=initializer_weights,
            bias_initializer=initializer_bias,
        )(x)
    else:
        output = Dense(
            output_dim,
            activation=out_activation,
            kernel_initializer=initializer_weights,
            bias_initializer=initializer_bias,
        )(inputs)

    model = Model(inputs=inputs, outputs=output)

    model.summary()

    kerasDefaults = candle.keras_default_config()
    if params["momentum"]:
        kerasDefaults["momentum_sgd"] = params["momentum"]

    optimizer = candle.build_optimizer(
        params["optimizer"], params["learning_rate"], kerasDefaults
    )

    model.compile(
        loss=params["loss"], optimizer=optimizer, metrics=["mae", r2],
    )

    # set up a bunch of callbacks to do work during model training..

    checkpointer = ModelCheckpoint(
        filepath=params["save_path"] + "agg_adrp.autosave.model.h5",
        verbose=1,
        save_weights_only=False,
        save_best_only=True,
    )
    csv_logger = CSVLogger(params["save_path"] + "agg_adrp.training.log")

    min_lr = params['learning_rate']*params['reduce_ratio']
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.75,
        patience=params['reduce_patience'],
        mode="auto",
        verbose=1,
        epsilon=0.0001,
        cooldown=3,
        min_lr=min_lr
    )

    early_stop = EarlyStopping(monitor="val_loss", 
                               patience=params['early_patience'],
                               verbose=1, 
                               mode="auto")

    # history = parallel_model.fit(X_train, Y_train,
    epochs = params["epochs"]
    batch_size = params["batch_size"]
    timeout_monitor = candle.TerminateOnTimeOut(params['timeout'])
    
    history = model.fit(
        X_train,
        Y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        validation_data=(X_test, Y_test),
        callbacks=[checkpointer, timeout_monitor, csv_logger, reduce_lr, early_stop],
    )

    print("Reloading saved best model")
    model.load_weights(params['save_path'] + "agg_adrp.autosave.model.h5")

    score = model.evaluate(X_test, Y_test, verbose=0)

    print(score)

    print(history.history.keys())

    # see big fuction below, creates plots etc.
    # TODO: Break post_process into multiple functions
    post_process(params, X_train, X_test, Y_test, score, history, model)

    adrp.logger.handlers = []

    return history


def post_process(params, X_train, X_test, Y_test, score, history, model):
    save_path = params["save_path"]
    print("saving to path: ", save_path)

    # summarize history for MAE
    plt.plot(history.history["mae"])
    plt.plot(history.history["val_mae"])
    plt.title("Model Mean Absolute Error")
    plt.ylabel("mae")
    plt.xlabel("epoch")
    plt.legend(["train", "test"], loc="upper left")

    plt.savefig(save_path + "agg_adrp.mae.png", bbox_inches="tight")
    plt.savefig(save_path + "agg_adrp.mae.pdf", bbox_inches="tight")

    plt.close()

    # summarize history for loss
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("Model Loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "test"], loc="upper left")

    plt.savefig(save_path + "agg_adrp.loss.png", bbox_inches="tight")
    plt.savefig(save_path + "agg_adrp.loss.pdf", bbox_inches="tight")

    plt.close()

    print("Test val_loss:", score[0])
    print("Test val_mae:", score[1])

    # serialize model to JSON
    model_json = model.to_json()
    with open(save_path + "agg_adrp.model.json", "w") as json_file:
        json_file.write(model_json)

    # serialize model to YAML
    model_yaml = model.to_yaml()
    with open(save_path + "agg_adrp.model.yaml", "w") as yaml_file:
        yaml_file.write(model_yaml)

    # serialize weights to HDF5
    model.save_weights(save_path + "agg_adrp.model.h5")
    print("Saved model to disk")

    # load json and create model
    json_file = open(save_path + "agg_adrp.model.json", "r")
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model_json = model_from_json(loaded_model_json)

    # load yaml and create model
    yaml_file = open(save_path + "agg_adrp.model.yaml", "r")
    loaded_model_yaml = yaml_file.read()
    yaml_file.close()
    loaded_model_yaml = model_from_yaml(loaded_model_yaml)

    # load weights into new model
    loaded_model_json.load_weights(save_path + "agg_adrp.model.h5")
    print("Loaded json model from disk")

    # evaluate json loaded model on test data
    loaded_model_json.compile(
        loss="binary_crossentropy", optimizer="SGD", metrics=["mean_absolute_error"]
    )
    score_json = loaded_model_json.evaluate(X_test, Y_test, verbose=0)

    print("json Validation loss:", score_json[0])
    print("json Validation mae:", score_json[1])

    print("json %s: %.2f%%" % (loaded_model_json.metrics_names[1], score_json[1] * 100))

    # load weights into new model
    loaded_model_yaml.load_weights(save_path + "agg_adrp.model.h5")
    print("Loaded yaml model from disk")

    # evaluate loaded model on test data
    loaded_model_yaml.compile(
        loss="binary_crossentropy", optimizer="SGD", metrics=["mean_absolute_error"]
    )
    score_yaml = loaded_model_yaml.evaluate(X_test, Y_test, verbose=0)

    print("yaml Validation loss:", score_yaml[0])
    print("yaml Validation mae:", score_yaml[1])

    print("yaml %s: %.2f%%" % (loaded_model_yaml.metrics_names[1], score_yaml[1] * 100))

    # predict using loaded yaml model on test and training data

    predict_yaml_train = loaded_model_yaml.predict(X_train)

    predict_yaml_test = loaded_model_yaml.predict(X_test)

    print("Yaml_train_shape:", predict_yaml_train.shape)
    print("Yaml_test_shape:", predict_yaml_test.shape)

    predict_yaml_train_classes = np.argmax(predict_yaml_train, axis=1)
    predict_yaml_test_classes = np.argmax(predict_yaml_test, axis=1)

    np.savetxt(
        save_path + "predict_yaml_train.csv",
        predict_yaml_train,
        delimiter=",",
        fmt="%.3f",
    )
    np.savetxt(
        save_path + "predict_yaml_test.csv",
        predict_yaml_test,
        delimiter=",",
        fmt="%.3f",
    )

    np.savetxt(
        save_path + "predict_yaml_train_classes.csv",
        predict_yaml_train_classes,
        delimiter=",",
        fmt="%d",
    )
    np.savetxt(
        save_path + "predict_yaml_test_classes.csv",
        predict_yaml_test_classes,
        delimiter=",",
        fmt="%d",
    )


def main():
    params = initialize_parameters()
    run(params)


if __name__ == "__main__":
    main()
    if K.backend() == "tensorflow":
        K.clear_session()
