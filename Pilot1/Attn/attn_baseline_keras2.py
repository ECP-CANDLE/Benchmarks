from __future__ import print_function

import itertools
import pandas as pd
import numpy as np
import os
import sys
import gzip
import argparse
import sklearn

import tensorflow as tf

import keras as ke
from keras import backend as K

from keras.layers import Input, Dense, Dropout, Activation, BatchNormalization
from keras.optimizers import SGD, Adam, RMSprop, Adadelta
from keras.models import Sequential, Model, model_from_json, model_from_yaml
from keras.utils import np_utils, multi_gpu_model

from keras.callbacks import Callback, ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping, TensorBoard

from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, roc_auc_score, confusion_matrix, balanced_accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler
from sklearn.metrics import recall_score, auc, roc_curve, f1_score, precision_recall_curve

import attn
import candle

import attn_viz_utils as attnviz

np.set_printoptions(precision=4)


def r2(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return (1 - SS_res / (SS_tot + K.epsilon()))


def tf_auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc


def auroc(y_true, y_pred):
    score = tf.py_func(lambda y_true, y_pred: roc_auc_score(y_true, y_pred, average='macro', sample_weight=None).astype('float32'),
                       [y_true, y_pred],
                       'float32',
                       stateful=False,
                       name='sklearnAUC')
    return score


def covariance(x, y):
    return K.mean(x * y) - K.mean(x) * K.mean(y)


def corr(y_true, y_pred):
    cov = covariance(y_true, y_pred)
    var1 = covariance(y_true, y_true)
    var2 = covariance(y_pred, y_pred)
    return cov / (K.sqrt(var1 * var2) + K.epsilon())


def xent(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred)


def mse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred)


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
        msg = "[Epoch: %i] %s" % (epoch, ", ".join("%s: %f" % (k, v) for k, v in sorted(logs.items())))
        self.print_fcn(msg)


def build_type_classifier(x_train, y_train, x_test, y_test):
    y_train = np.argmax(y_train, axis=1)
    y_test = np.argmax(y_test, axis=1)
    from xgboost import XGBClassifier
    clf = XGBClassifier(max_depth=6, n_estimators=100)
    clf.fit(x_train, y_train, eval_set=[(x_train, y_train), (x_test, y_test)], verbose=False)
    y_pred = clf.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    print(acc)
    return clf


def initialize_parameters(default_model='attn_default_model.txt'):

    # Build benchmark object
    attnBmk = attn.BenchmarkAttn(attn.file_path, default_model, 'keras',
                                 prog='attn_baseline',
                                 desc='Multi-task (DNN) for data extraction from clinical reports - Pilot 3 Benchmark 1')

    # Initialize parameters
    gParameters = candle.finalize_parameters(attnBmk)

    return gParameters


def save_cache(cache_file, x_train, y_train, x_val, y_val, x_test, y_test, x_labels, y_labels):
    with h5py.File(cache_file, 'w') as hf:
        hf.create_dataset("x_train", data=x_train)
        hf.create_dataset("y_train", data=y_train)
        hf.create_dataset("x_val", data=x_val)
        hf.create_dataset("y_val", data=y_val)
        hf.create_dataset("x_test", data=x_test)
        hf.create_dataset("y_test", data=y_test)
        hf.create_dataset("x_labels", (len(x_labels), 1), 'S100', data=[x.encode("ascii", "ignore") for x in x_labels])
        hf.create_dataset("y_labels", (len(y_labels), 1), 'S100', data=[x.encode("ascii", "ignore") for x in y_labels])


def load_cache(cache_file):
    with h5py.File(cache_file, 'r') as hf:
        x_train = hf['x_train'][:]
        y_train = hf['y_train'][:]
        x_val = hf['x_val'][:]
        y_val = hf['y_val'][:]
        x_test = hf['x_test'][:]
        y_test = hf['y_test'][:]
        x_labels = [x[0].decode('unicode_escape') for x in hf['x_labels'][:]]
        y_labels = [x[0].decode('unicode_escape') for x in hf['y_labels'][:]]
    return x_train, y_train, x_val, y_val, x_test, y_test, x_labels, y_labels


def build_attention_model(params, PS):

    assert (len(params['dense']) == len(params['activation']))
    assert (len(params['dense']) > 3)

    DR = params['dropout']
    inputs = Input(shape=(PS,))
    x = Dense(params['dense'][0], activation=params['activation'][0])(inputs)
    x = BatchNormalization()(x)
    a = Dense(params['dense'][1], activation=params['activation'][1])(x)
    a = BatchNormalization()(a)
    b = Dense(params['dense'][2], activation=params['activation'][2])(x)
    x = ke.layers.multiply([a, b])

    for i in range(3, len(params['dense']) - 1):
        x = Dense(params['dense'][i], activation=params['activation'][i])(x)
        x = BatchNormalization()(x)
        x = Dropout(DR)(x)

    outputs = Dense(params['dense'][-1], activation=params['activation'][-1])(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.summary()

    return model


def run(params):
    args = candle.ArgumentStruct(**params)
    seed = args.rng_seed
    candle.set_seed(seed)

    # Construct extension to save model
    ext = attn.extension_from_parameters(params, 'keras')
    candle.verify_path(params['save_path'])
    prefix = '{}{}'.format(params['save_path'], ext)
    logfile = params['logfile'] if params['logfile'] else prefix + '.log'
    root_fname = 'Agg_attn_bin'
    candle.set_up_logger(logfile, attn.logger, params['verbose'])
    attn.logger.info('Params: {}'.format(params))

    # Get default parameters for initialization and optimizer functions
    keras_defaults = candle.keras_default_config()

    ##
    X_train, _Y_train, X_val, _Y_val, X_test, _Y_test = attn.load_data(params, seed)

    # move this inside the load_data function
    Y_train = _Y_train['AUC']
    Y_test = _Y_test['AUC']
    Y_val = _Y_val['AUC']

    Y_train_neg, Y_train_pos = np.bincount(Y_train)
    Y_test_neg, Y_test_pos = np.bincount(Y_test)
    Y_val_neg, Y_val_pos = np.bincount(Y_val)

    Y_train_total = Y_train_neg + Y_train_pos
    Y_test_total = Y_test_neg + Y_test_pos
    Y_val_total = Y_val_neg + Y_val_pos

    total = Y_train_total + Y_test_total + Y_val_total
    neg = Y_train_neg + Y_test_neg + Y_val_neg
    pos = Y_train_pos + Y_test_pos + Y_val_pos

    print('Examples:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n'.format(
        total, pos, 100 * pos / total))

    nb_classes = params['dense'][-1]

    Y_train = np_utils.to_categorical(Y_train, nb_classes)
    Y_test = np_utils.to_categorical(Y_test, nb_classes)
    Y_val = np_utils.to_categorical(Y_val, nb_classes)

    y_integers = np.argmax(Y_train, axis=1)
    class_weights = compute_class_weight('balanced', np.unique(y_integers), y_integers)
    d_class_weights = dict(enumerate(class_weights))

    print('X_train shape:', X_train.shape)
    print('X_test shape:', X_test.shape)

    print('Y_train shape:', Y_train.shape)
    print('Y_test shape:', Y_test.shape)

    PS = X_train.shape[1]
    model = build_attention_model(params, PS)

    # parallel_model = multi_gpu_model(model, gpus=4)
    # parallel_model.compile(loss='mean_squared_error',
    #                        optimizer=SGD(lr=0.0001, momentum=0.9),
    #                        metrics=['mae',r2])
    kerasDefaults = candle.keras_default_config()
    if params['momentum']:
        kerasDefaults['momentum_sgd'] = params['momentum']

    optimizer = candle.build_optimizer(params['optimizer'], params['learning_rate'], kerasDefaults)

    model.compile(loss=params['loss'],
                  optimizer=optimizer,
                  #           SGD(lr=0.00001, momentum=0.9),
                  # optimizer=Adam(lr=0.00001),
                  # optimizer=RMSprop(lr=0.0001),
                  # optimizer=Adadelta(),
                  metrics=['acc', tf_auc])

    # set up a bunch of callbacks to do work during model training..

    checkpointer = ModelCheckpoint(filepath=params['save_path'] + root_fname + '.autosave.model.h5', verbose=1, save_weights_only=False, save_best_only=True)
    csv_logger = CSVLogger('{}/{}.training.log'.format(params['save_path'], root_fname))
    reduce_lr = ReduceLROnPlateau(monitor='val_tf_auc', factor=0.20, patience=40, verbose=1, mode='auto', min_delta=0.0001, cooldown=3, min_lr=0.000000001)
    early_stop = EarlyStopping(monitor='val_tf_auc', patience=200, verbose=1, mode='auto')
    candle_monitor = candle.CandleRemoteMonitor(params=params)

    candle_monitor = candle.CandleRemoteMonitor(params=params)
    timeout_monitor = candle.TerminateOnTimeOut(params['timeout'])
    tensorboard = TensorBoard(log_dir="tb/tb{}".format(ext))

    history_logger = LoggingCallback(attn.logger.debug)

    callbacks = [candle_monitor, timeout_monitor, csv_logger, history_logger]

    if params['reduce_lr']:
        callbacks.append(reduce_lr)

    if params['use_cp']:
        callbacks.append(checkpointer)
    if params['use_tb']:
        callbacks.append(tensorboard)
    if params['early_stop']:
        callbacks.append(early_stop)

    epochs = params['epochs']
    batch_size = params['batch_size']
    history = model.fit(X_train, Y_train, class_weight=d_class_weights,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(X_val, Y_val),
                        callbacks=callbacks)

    # diagnostic plots
    if 'loss' in history.history.keys():
        candle.plot_history(params['save_path'] + root_fname, history, 'loss')
    if 'acc' in history.history.keys():
        candle.plot_history(params['save_path'] + root_fname, history, 'acc')
    if 'tf_auc' in history.history.keys():
        candle.plot_history(params['save_path'] + root_fname, history, 'tf_auc')

    # Evaluate model
    score = model.evaluate(X_test, Y_test, verbose=0)
    Y_predict = model.predict(X_test)

    evaluate_model(params, root_fname, nb_classes, Y_test, _Y_test, Y_predict, pos, total, score)

    save_and_test_saved_model(params, model, root_fname, X_train, X_test, Y_test)

    attn.logger.handlers = []

    return history


def evaluate_model(params, root_fname, nb_classes, Y_test, _Y_test, Y_predict, pos, total, score):

    threshold = 0.5

    Y_pred_int = (Y_predict[:, 0] < threshold).astype(np.int)
    Y_test_int = (Y_test[:, 0] < threshold).astype(np.int)

    print('creating table of predictions')
    f = open(params['save_path'] + root_fname + '.predictions.tsv', 'w')
    for index, row in _Y_test.iterrows():
        if row['AUC'] == 1:
            if Y_pred_int[index] == 1:
                call = 'TP'
            else:
                call = 'FN'
        if row['AUC'] == 0:
            if Y_pred_int[index] == 0:
                call = 'TN'
            else:
                call = 'FP'
        # 1 TN 0 0.6323 NCI60.786-0 NSC.256439 NSC.102816
        print(index, "\t", call, "\t", Y_pred_int[index], "\t", row['AUC'], "\t", row['Sample'], "\t", row['Drug1'], file=f)
    f.close()

    false_pos_rate, true_pos_rate, thresholds = roc_curve(Y_test[:, 0], Y_predict[:, 0])
    roc_auc = auc(false_pos_rate, true_pos_rate)

    auc_keras = roc_auc
    fpr_keras = false_pos_rate
    tpr_keras = true_pos_rate

    # ROC plots
    fname = params['save_path'] + root_fname + '.auroc.pdf'
    print('creating figure at ', fname)
    attnviz.plot_ROC(fpr_keras, tpr_keras, auc_keras, fname)
    # Zoom in view of the upper left corner.
    fname = params['save_path'] + root_fname + '.auroc_zoom.pdf'
    print('creating figure at ', fname)
    attnviz.plot_ROC(fpr_keras, tpr_keras, auc_keras, fname, zoom=True)

    f1 = f1_score(Y_test_int, Y_pred_int)

    precision, recall, thresholds = precision_recall_curve(Y_test[:, 0], Y_predict[:, 0])
    pr_auc = auc(recall, precision)

    pr_keras = pr_auc
    precision_keras = precision
    recall_keras = recall

    print('f1=%.3f auroc=%.3f aucpr=%.3f' % (f1, auc_keras, pr_keras))
    # Plot RF
    fname = params['save_path'] + root_fname + '.aurpr.pdf'
    print('creating figure at ', fname)
    no_skill = len(Y_test_int[Y_test_int == 1]) / len(Y_test_int)
    attnviz.plot_RF(recall_keras, precision_keras, pr_keras, no_skill, fname)

    # Compute confusion matrix
    cnf_matrix = sklearn.metrics.confusion_matrix(Y_test_int, Y_pred_int)
    # Plot non-normalized confusion matrix
    class_names = ["Non-Response", "Response"]
    fname = params['save_path'] + root_fname + '.confusion_without_norm.pdf'
    print('creating figure at ', fname)
    attnviz.plot_confusion_matrix(cnf_matrix, fname, classes=class_names, title='Confusion matrix, without normalization')
    # Plot normalized confusion matrix
    fname = params['save_path'] + root_fname + '.confusion_with_norm.pdf'
    print('creating figure at ', fname)
    attnviz.plot_confusion_matrix(cnf_matrix, fname, classes=class_names, normalize=True, title='Normalized confusion matrix')

    print('Examples:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n'.format(total, pos, 100 * pos / total))

    print(sklearn.metrics.roc_auc_score(Y_test_int, Y_pred_int))
    print(sklearn.metrics.balanced_accuracy_score(Y_test_int, Y_pred_int))
    print(sklearn.metrics.classification_report(Y_test_int, Y_pred_int))
    print(sklearn.metrics.confusion_matrix(Y_test_int, Y_pred_int))
    print("score")
    print(score)

    print('Test val_loss:', score[0])
    print('Test accuracy:', score[1])


def save_and_test_saved_model(params, model, root_fname, X_train, X_test, Y_test):

    # serialize model to JSON
    model_json = model.to_json()
    with open(params['save_path'] + root_fname + ".model.json", "w") as json_file:
        json_file.write(model_json)

    # serialize model to YAML
    model_yaml = model.to_yaml()
    with open(params['save_path'] + root_fname + ".model.yaml", "w") as yaml_file:
        yaml_file.write(model_yaml)

    # serialize weights to HDF5
    model.save_weights(params['save_path'] + root_fname + ".model.h5")
    print("Saved model to disk")

    # load json and create model
    json_file = open(params['save_path'] + root_fname + '.model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model_json = model_from_json(loaded_model_json)

    # load yaml and create model
    yaml_file = open(params['save_path'] + root_fname + '.model.yaml', 'r')
    loaded_model_yaml = yaml_file.read()
    yaml_file.close()
    loaded_model_yaml = model_from_yaml(loaded_model_yaml)

    # load weights into new model
    loaded_model_json.load_weights(params['save_path'] + root_fname + ".model.h5")
    print("Loaded json model from disk")

    # evaluate json loaded model on test data
    loaded_model_json.compile(loss='binary_crossentropy', optimizer=params['optimizer'], metrics=['accuracy'])
    score_json = loaded_model_json.evaluate(X_test, Y_test, verbose=0)

    print('json Validation loss:', score_json[0])
    print('json Validation accuracy:', score_json[1])

    print("json %s: %.2f%%" % (loaded_model_json.metrics_names[1], score_json[1] * 100))

    # load weights into new model
    loaded_model_yaml.load_weights(params['save_path'] + root_fname + ".model.h5")
    print("Loaded yaml model from disk")

    # evaluate loaded model on test data
    loaded_model_yaml.compile(loss='binary_crossentropy', optimizer=params['optimizer'], metrics=['accuracy'])
    score_yaml = loaded_model_yaml.evaluate(X_test, Y_test, verbose=0)
    print('yaml Validation loss:', score_yaml[0])
    print('yaml Validation accuracy:', score_yaml[1])
    print("yaml %s: %.2f%%" % (loaded_model_yaml.metrics_names[1], score_yaml[1] * 100))

    # predict using loaded yaml model on test and training data
    predict_yaml_train = loaded_model_yaml.predict(X_train)
    predict_yaml_test = loaded_model_yaml.predict(X_test)
    print('Yaml_train_shape:', predict_yaml_train.shape)
    print('Yaml_test_shape:', predict_yaml_test.shape)

    predict_yaml_train_classes = np.argmax(predict_yaml_train, axis=1)
    predict_yaml_test_classes = np.argmax(predict_yaml_test, axis=1)
    np.savetxt(params['save_path'] + root_fname + "_predict_yaml_train.csv", predict_yaml_train, delimiter=",", fmt="%.3f")
    np.savetxt(params['save_path'] + root_fname + "_predict_yaml_test.csv", predict_yaml_test, delimiter=",", fmt="%.3f")

    np.savetxt(params['save_path'] + root_fname + "_predict_yaml_train_classes.csv", predict_yaml_train_classes, delimiter=",", fmt="%d")
    np.savetxt(params['save_path'] + root_fname + "_predict_yaml_test_classes.csv", predict_yaml_test_classes, delimiter=",", fmt="%d")


def main():
    params = initialize_parameters()
    run(params)


if __name__ == '__main__':
    main()
    if K.backend() == 'tensorflow':
        K.clear_session()
