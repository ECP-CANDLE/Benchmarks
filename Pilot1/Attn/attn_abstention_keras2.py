from __future__ import print_function

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

from attn_baseline_keras2 import build_attention_model

np.set_printoptions(precision=4)

additional_definitions = [
    {'name': 'alpha_scale_factor',
     'type': float,
     'default': 0.8,
     'help': 'factor to increase or decrease weight for abstention term in cost function'},
    {'name': 'min_abs_acc',
     'type': float,
     'default': 0.7,
     'help': 'min target abstention accuracy'},
    {'name': 'max_abs_frac',
      'type': float,
      'default': 0.7,
      'help': 'max target abstention fraction'},
    {'name': 'acc_gain',
      'type': float,
      'default': 5.0,
      'help': 'factor to weight accuracy when determining new alpha scale'},
    {'name': 'abs_gain',
      'type': float,
      'default': 1.0,
      'help': 'factor to weight abstention fraction when determining new alpha scale'},
]

required = [
    'activation',
    'batch_size',
    'dense',
    'dropout',
    'epochs',
    'learning_rate',
    'loss',
    'optimizer',
    'rng_seed',
    'val_split',
    'timeout',
    'min_abs_acc',
    'max_abs_frac'
]


class BenchmarkAttnAbs(candle.Benchmark):
    def set_locals(self):
        """Functionality to set variables specific for the benchmark
        - required: set of required parameters for the benchmark.
        - additional_definitions: list of dictionaries describing the additional parameters for the
        benchmark.
        """

        if required is not None:
            self.required = set(required)
        if additional_definitions is not None:
            self.additional_definitions = additional_definitions + attn.additional_definitions


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


def initialize_parameters(default_model='attn_abs_default_model.txt'):

    # Build benchmark object
    attnAbsBmk = BenchmarkAttnAbs(
        attn.file_path, default_model, 'keras',
        prog='attention_abstention', desc='Attention model with abstention - Pilot 1 Benchmark')

    # Initialize parameters
    gParameters = candle.finalize_parameters(attnAbsBmk)

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


def extension_from_parameters(params, framework=''):
    """Construct string for saving model with annotation of parameters"""
    ext = framework + '.abs'
    for i, n in enumerate(params['dense']):
        if n:
            ext += '.D{}={}'.format(i + 1, n)
    ext += '.A={}'.format(params['activation'][0])
    ext += '.B={}'.format(params['batch_size'])
    ext += '.E={}'.format(params['epochs'])
    ext += '.LR={}'.format(params['learning_rate'])

    if params['dropout']:
        ext += '.DR={}'.format(params['dropout'])
    if params['warmup_lr']:
        ext += '.WU_LR'
    if params['reduce_lr']:
        ext += '.Re_LR'
    if params['residual']:
        ext += '.Res'

    return ext


def run(params):
    args = candle.ArgumentStruct(**params)
    seed = args.rng_seed
    candle.set_seed(seed)

    # Construct extension to save model
    ext = extension_from_parameters(params, 'keras')
    candle.verify_path(params['save_path'])
    prefix = '{}{}'.format(params['save_path'], ext)
    logfile = params['logfile'] if params['logfile'] else prefix + '.log'
    root_fname = 'Agg_attn_abs_bin'
    candle.set_up_logger(logfile, attn.logger, params['verbose'])
    attn.logger.info('Params: {}'.format(params))

    # Get default parameters for initialization and optimizer functions
    keras_defaults = candle.keras_default_config()

    ##
    X_train, _Y_train, X_val, _Y_val, X_test, _Y_test  = attn.load_data(params, seed)

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

    # Convert classes to categorical with an extra slot for the abstaining class
    Y_train, Y_test, Y_val = candle.modify_labels(nb_classes + 1, Y_train, Y_test, Y_val)

    # Try class weight and abstention classifier
    y_integers = np.argmax(Y_train, axis=1)
    class_weights = compute_class_weight('balanced', np.unique(y_integers), y_integers)
    d_class_weights = dict(enumerate(class_weights))

    print('X_train shape:', X_train.shape)
    print('X_test shape:', X_test.shape)

    print('Y_train shape:', Y_train.shape)
    print('Y_test shape:', Y_test.shape)

    PS = X_train.shape[1]
    model = build_attention_model(params, PS)
    model = candle.add_model_output(model, mode='abstain', num_add=1, activation='sigmoid')
    print('Model after modifying layer for abstention')
    model.summary()

    # Configure abstention model
    mask = np.zeros(nb_classes + 1)
    mask[-1] = 1
    alpha0 = 0.5  # In the long term this is not as important since alpha auto tunes, however it may require a large number of epochs to converge if set far away from target
    abstention_cbk = candle.AbstentionAdapt_Callback(acc_monitor='val_abstention_acc', 
                                                     abs_monitor='val_abstention', 
                                                     alpha0=alpha0, 
                                                     alpha_scale_factor=params['alpha_scale_factor'], 
                                                     min_abs_acc=params['min_abs_acc'],
                                                     max_abs_frac=params['max_abs_frac'],
                                                     acc_gain=params['acc_gain'],
                                                     abs_gain=params['abs_gain'],
                                                     )

    # parallel_model = multi_gpu_model(model, gpus=4)
    # parallel_model.compile(loss='mean_squared_error',
    #                        optimizer=SGD(lr=0.0001, momentum=0.9),
    #                        metrics=['mae',r2])
    kerasDefaults = candle.keras_default_config()
    if params['momentum']:
        kerasDefaults['momentum_sgd'] = params['momentum']

    optimizer = candle.build_optimizer(params['optimizer'], params['learning_rate'], kerasDefaults)

    # compile model with abstention loss
    model.compile(
        loss=candle.abstention_loss(abstention_cbk.alpha, mask),
        optimizer=optimizer,
        metrics=['acc', tf_auc, 
                 candle.abstention_acc_metric(nb_classes), 
                 candle.acc_class_i_metric(1), 
                 candle.abstention_acc_class_i_metric(nb_classes, 1),
                 candle.abstention_metric(nb_classes)])

    # set up a bunch of callbacks to do work during model training..
    checkpointer = ModelCheckpoint(filepath=params['save_path'] + root_fname + '.autosave.model.h5', verbose=1, save_weights_only=False, save_best_only=True)
    csv_logger = CSVLogger('{}/{}.training.log'.format(params['save_path'], root_fname))
    reduce_lr = ReduceLROnPlateau(monitor='val_tf_auc', factor=0.20, patience=40, verbose=1, mode='auto', min_delta=0.0001, cooldown=3, min_lr=0.000000001)
    early_stop = EarlyStopping(monitor='val_tf_auc', patience=200, verbose=1, mode='auto')
    candle_monitor = candle.CandleRemoteMonitor(params=params)

    candle_monitor = candle.CandleRemoteMonitor(params=params)
    timeout_monitor = candle.TerminateOnTimeOut(params['timeout'])
    tensorboard = TensorBoard(log_dir="tb/tb{}".format(ext))

    history_logger = candle.LoggingCallback(attn.logger.debug)

    # abstention_cbk = candle.AbstentionAdapt_Callback(monitor='val_abs_acc_class1', abs_monitor='val_abstention', scale_factor=params['abs_scale_factor'], target_acc=params['target_abs_acc'])

    callbacks = [candle_monitor, timeout_monitor, csv_logger, history_logger, abstention_cbk]

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
    history = model.fit(X_train, Y_train,  class_weight=d_class_weights,
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
    if 'abstention_acc' in history.history.keys():
        candle.plot_history(params['save_path'] + root_fname, history, 'abstention_acc')
    # Plot alpha evolution
    fname = params['save_path'] + root_fname + '.alpha.png'
    xlabel = 'Epochs'
    ylabel = 'Abstention Weight alpha'
    title = 'alpha Evolution'
    candle.plot_array(abstention_cbk.alphavalues, xlabel, ylabel, title, fname)

    # Evaluate model
    score = model.evaluate(X_test, Y_test, verbose=0)
    Y_predict = model.predict(X_test)
    evaluate_abstention(params, root_fname, nb_classes, Y_test, _Y_test, Y_predict, pos, total, score)

    save_and_test_saved_model(params, model, root_fname, nb_classes, abstention_cbk.alpha, mask, X_train, X_test, Y_test)

    attn.logger.handlers = []
    df_testX = pd.DataFrame(X_test)
    #print('df_testX.shape: ', df_testX.shape)
    cols = ['Y_test' + str(i) for i in range(Y_test.shape[1])]
    df_testY = pd.DataFrame(Y_test, columns=cols)
    #print('df_testY.shape: ', df_testY.shape)
    df_test = pd.concat([df_testY, df_testX], axis=1)
    #print('df_test.shape: ', df_test.shape)
    cols = ['Y_pred' + str(i) for i in range(Y_predict.shape[1])]
    df_pred = pd.DataFrame(Y_predict, columns=cols)
    #print('df_pred.shape: ', df_pred.shape)
    df_test = pd.concat([df_pred, df_test], axis=1).reset_index(drop=True)
    #print('df_test.shape: ', df_test.shape)
    fname = params['save_path'] + root_fname + '.dftest.tsv'
    df_test.to_csv(fname, sep='\t', index=False, float_format="%.3g")


    return history


def evaluate_abstention(params, root_fname, nb_classes, Y_test, _Y_test, Y_predict, pos, total, score):
    Y_pred_int  = np.argmax(Y_predict, axis=1).astype(np.int)
    Y_test_int  = np.argmax(Y_test, axis=1).astype(np.int)

    # Get samples where it abstains from predicting
    Y_pred_abs = (Y_pred_int == nb_classes).astype(np.int)

    abs0 = 0
    abs1 = 0
    print('creating table of predictions (with abstention)')
    f = open(params['save_path'] + root_fname + '.predictions.tsv', 'w')

    for index, row in _Y_test.iterrows():

        if row['AUC'] == 1:
            if Y_pred_abs[index] == 1:  # abstaining in this sample
                call = 'ABS1'
                abs1 += 1
            else:  # Prediction is made (no abstention)
                if Y_pred_int[index] == 1:
                    call = 'TP'
                else:
                    call = 'FN'
        if row['AUC'] == 0:
            if Y_pred_abs[index] == 1:  # abstaining in this sample
                call = 'ABS0'
                abs0 += 1
            else:  # Prediction is made (no abstention)
                if Y_pred_int[index] == 0:
                    call = 'TN'
                else:
                    call = 'FP'

        print(index, "\t", call, "\t", Y_pred_int[index], "\t", row['AUC'], "\t", Y_pred_abs[index], "\t", row['Sample'], "\t", row['Drug1'], file=f)

    f.close()

    # Filtering samples by predictions made (i.e. leave just the predicted samples where there is NO abstention)
    index_pred_noabs = (Y_pred_int < nb_classes)
    Y_test_noabs = Y_test[index_pred_noabs, :2]
    Y_test_int_noabs = Y_test_int[index_pred_noabs]
    Y_pred_noabs = Y_predict[index_pred_noabs, :2] / np.sum(Y_predict[index_pred_noabs, :2], axis=1, keepdims=True)
    Y_pred_int_noabs = Y_pred_int[index_pred_noabs]
    false_pos_rate, true_pos_rate, thresholds = roc_curve(Y_test_noabs[:, 0], Y_pred_noabs[:, 0])

    roc_auc = auc(false_pos_rate, true_pos_rate)

    auc_keras = roc_auc
    fpr_keras = false_pos_rate
    tpr_keras = true_pos_rate

    # ROC plots
    fname = params['save_path'] + root_fname + '.auroc.pdf'
    print('creating figure at ', fname)
    add_lbl = ' (after removing abstained samples) '
    attnviz.plot_ROC(fpr_keras, tpr_keras, auc_keras, fname, xlabel_add=add_lbl, ylabel_add=add_lbl)
    # Zoom in view of the upper left corner.
    fname = params['save_path'] + root_fname + '.auroc_zoom.pdf'
    print('creating figure at ', fname)
    attnviz.plot_ROC(fpr_keras, tpr_keras, auc_keras, fname, xlabel_add=add_lbl, ylabel_add=add_lbl, zoom=True)

    f1 = f1_score(Y_test_int_noabs, Y_pred_int_noabs)
    precision, recall, thresholds = precision_recall_curve(Y_test_noabs[:, 0], Y_pred_noabs[:, 0])
    pr_auc = auc(recall, precision)
    pr_keras = pr_auc
    precision_keras = precision
    recall_keras = recall
    print('f1=%.3f auroc=%.3f aucpr=%.3f' % (f1, auc_keras, pr_keras))
    # Plot RF
    fname = params['save_path'] + root_fname + '.aurpr.pdf'
    print('creating figure at ', fname)
    no_skill = len(Y_test_int_noabs[Y_test_int_noabs == 1]) / len(Y_test_int_noabs)
    attnviz.plot_RF(recall_keras, precision_keras, pr_keras, no_skill, fname, xlabel_add=add_lbl, ylabel_add=add_lbl)

    # Compute confusion matrix (complete)
    cnf_matrix = sklearn.metrics.confusion_matrix(Y_test_int, Y_pred_int)
    # Plot non-normalized confusion matrix
    class_names = ['Non-Response', 'Response', 'Abstain']
    fname = params['save_path'] + root_fname + '.confusion_without_norm.pdf'
    attnviz.plot_confusion_matrix(cnf_matrix, fname, classes=class_names, title='Confusion matrix, without normalization')
    print('NOTE: Confusion matrix above has zeros in the last row since the ground-truth does not include samples in the abstaining class.')
    # Plot normalized confusion matrix
    fname = params['save_path'] + root_fname + '.confusion_with_norm.pdf'
    attnviz.plot_confusion_matrix(cnf_matrix, fname, classes=class_names, normalize=True, title='Normalized confusion matrix')
    print('NOTE: Normalized confusion matrix above has NaNs in the last row since the ground-truth does not include samples in the abstaining class.')

    print('Examples:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n'.format(total, pos, 100 * pos / total))
    total_pred = Y_pred_int_noabs.shape[0]
    print('Abstention (in prediction):  Label0: {} ({:.2f}% of total pred)\n Label1: {} ({:.2f}% of total pred)\n'.format(abs0, 100 * abs0 / total_pred, abs1, 100 * abs1 / total_pred))
    print(sklearn.metrics.roc_auc_score(Y_test_int_noabs, Y_pred_int_noabs))
    print(sklearn.metrics.balanced_accuracy_score(Y_test_int_noabs, Y_pred_int_noabs))
    print(sklearn.metrics.classification_report(Y_test_int_noabs, Y_pred_int_noabs))
    print(sklearn.metrics.confusion_matrix(Y_test_int_noabs, Y_pred_int_noabs))
    print('Score: ', score)
    print('Test val_loss (not abstained samples):', score[0])
    print('Test accuracy (not abstained samples):', score[1])


def save_and_test_saved_model(params, model, root_fname, nb_classes, alpha, mask, X_train, X_test, Y_test):

    # serialize model to JSON
    model_json = model.to_json()
    with open(params['save_path'] + root_fname + '.model.json', "w") as json_file:
        json_file.write(model_json)

    # serialize model to YAML
    model_yaml = model.to_yaml()
    with open(params['save_path'] + root_fname + '.model.yaml', "w") as yaml_file:

        yaml_file.write(model_yaml)

    # serialize weights to HDF5
    model.save_weights(params['save_path'] + root_fname + '.model.h5')
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
    # yaml.load(input, Loader=yaml.FullLoader)

    # load weights into new model
    loaded_model_json.load_weights(params['save_path'] + root_fname + '.model.h5')
    # input = params['save_path'] + root_fname +  '.model.h5'
    # loaded_model_json.load(input, Loader=yaml.FullLoader)
    # print("Loaded json model from disk")

    # evaluate json loaded model on test data
    loaded_model_json.compile(loss=candle.abstention_loss(alpha, mask), optimizer='SGD', metrics=[candle.abstention_acc_metric(nb_classes)])
    score_json = loaded_model_json.evaluate(X_test, Y_test, verbose=0)
    print('json Validation abstention loss:', score_json[0])
    print('json Validation abstention accuracy:', score_json[1])
    print("json %s: %.2f%%" % (loaded_model_json.metrics_names[1], score_json[1] * 100))

    # load weights into new model
    loaded_model_yaml.load_weights(params['save_path'] + root_fname + '.model.h5')
    print("Loaded yaml model from disk")
    # evaluate yaml loaded model on test data
    loaded_model_yaml.compile(loss=candle.abstention_loss(alpha, mask), optimizer='SGD', metrics=[candle.abstention_acc_metric(nb_classes)])
    score_yaml = loaded_model_yaml.evaluate(X_test, Y_test, verbose=0)
    print('yaml Validation abstention loss:', score_yaml[0])
    print('yaml Validation abstention accuracy:', score_yaml[1])
    print("yaml %s: %.2f%%" % (loaded_model_yaml.metrics_names[1], score_yaml[1] * 100))

    # predict using loaded yaml model on test and training data
    predict_yaml_train = loaded_model_yaml.predict(X_train)
    predict_yaml_test = loaded_model_yaml.predict(X_test)
    print('Yaml_train_shape:', predict_yaml_train.shape)
    print('Yaml_test_shape:', predict_yaml_test.shape)
    predict_yaml_train_classes = np.argmax(predict_yaml_train, axis=1)
    predict_yaml_test_classes = np.argmax(predict_yaml_test, axis=1)
    np.savetxt(params['save_path'] + root_fname + '_predict_yaml_train.csv', predict_yaml_train, delimiter=",", fmt="%.3f")
    np.savetxt(params['save_path'] + root_fname + '_predict_yaml_test.csv', predict_yaml_test, delimiter=",", fmt="%.3f")
    np.savetxt(params['save_path'] + root_fname + '_predict_yaml_train_classes.csv', predict_yaml_train_classes, delimiter=",", fmt="%d")
    np.savetxt(params['save_path'] + root_fname + '_predict_yaml_test_classes.csv', predict_yaml_test_classes, delimiter=",", fmt="%d")


def main():
    params = initialize_parameters()
    run(params)


if __name__ == '__main__':
    main()
    if K.backend() == 'tensorflow':
        K.clear_session()
