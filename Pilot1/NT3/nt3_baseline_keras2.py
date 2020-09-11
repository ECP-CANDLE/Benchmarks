from __future__ import print_function

import pandas as pd
import numpy as np
import os
import sys
import gzip

from keras import backend as K

from keras.layers import Input, Dense, Dropout, Activation, Conv1D, MaxPooling1D, Flatten
from keras.optimizers import SGD, Adam, RMSprop
from keras.models import Sequential, Model, model_from_json, model_from_yaml
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler

import nt3 as bmk
import candle


def initialize_parameters(default_model='nt3_default_model.txt'):

    # Build benchmark object
    nt3Bmk = bmk.BenchmarkNT3(
        bmk.file_path,
        default_model,
        'keras',
        prog='nt3_baseline',
        desc='1D CNN to classify RNA sequence data in normal or tumor classes')

    # Initialize parameters
    gParameters = candle.finalize_parameters(nt3Bmk)

    return gParameters


def load_data(train_path, test_path, gParameters):

    print('Loading data...')
    df_train = (pd.read_csv(train_path, header=None).values).astype('float32')
    df_test = (pd.read_csv(test_path, header=None).values).astype('float32')
    print('done')

    print('df_train shape:', df_train.shape)
    print('df_test shape:', df_test.shape)

    seqlen = df_train.shape[1]

    df_y_train = df_train[:, 0].astype('int')
    df_y_test = df_test[:, 0].astype('int')

    # only training set has noise
    Y_train = np_utils.to_categorical(df_y_train, gParameters['classes'])
    Y_test = np_utils.to_categorical(df_y_test, gParameters['classes'])

    df_x_train = df_train[:, 1:seqlen].astype(np.float32)
    df_x_test = df_test[:, 1:seqlen].astype(np.float32)

    X_train = df_x_train
    X_test = df_x_test

    scaler = MaxAbsScaler()
    mat = np.concatenate((X_train, X_test), axis=0)
    mat = scaler.fit_transform(mat)

    X_train = mat[:X_train.shape[0], :]
    X_test = mat[X_train.shape[0]:, :]

    # TODO: Add better names for noise boolean, make a featue for both RNA seq and label noise together
    # check if noise is on (this is for label)
    if gParameters['add_noise']:
        # check if we want noise correlated with a feature
        if gParameters['noise_correlated']:
            Y_train, y_train_noise_gen = candle.label_flip_correlated(Y_train,
                                                                      gParameters['label_noise'], X_train,
                                                                      gParameters['feature_col'],
                                                                      gParameters['feature_threshold'])
        # else add uncorrelated noise
        else:
            Y_train, y_train_noise_gen = candle.label_flip(Y_train, gParameters['label_noise'])
    # check if noise is on for RNA-seq data
    elif gParameters['noise_gaussian']:
        X_train = candle.add_gaussian_noise(X_train, 0, gParameters['std_dev'])
	    
    return X_train, Y_train, X_test, Y_test


def run(gParameters):

    print('Params:', gParameters)

    file_train = gParameters['train_data']
    file_test = gParameters['test_data']
    url = gParameters['data_url']

    train_file = candle.get_file(file_train, url + file_train, cache_subdir='Pilot1')
    test_file = candle.get_file(file_test, url + file_test, cache_subdir='Pilot1')

    X_train, Y_train, X_test, Y_test = load_data(train_file, test_file, gParameters)

    print('X_train shape:', X_train.shape)
    print('X_test shape:', X_test.shape)

    print('Y_train shape:', Y_train.shape)
    print('Y_test shape:', Y_test.shape)

    x_train_len = X_train.shape[1]

    # this reshaping is critical for the Conv1D to work

    X_train = np.expand_dims(X_train, axis=2)
    X_test = np.expand_dims(X_test, axis=2)

    print('X_train shape:', X_train.shape)
    print('X_test shape:', X_test.shape)

    model = Sequential()

    layer_list = list(range(0, len(gParameters['conv']), 3))
    for _, i in enumerate(layer_list):
        filters = gParameters['conv'][i]
        filter_len = gParameters['conv'][i + 1]
        stride = gParameters['conv'][i + 2]
        print(int(i / 3), filters, filter_len, stride)
        if gParameters['pool']:
            pool_list = gParameters['pool']
            if type(pool_list) != list:
                pool_list = list(pool_list)

        if filters <= 0 or filter_len <= 0 or stride <= 0:
            break
        if 'locally_connected' in gParameters:
            model.add(LocallyConnected1D(filters, filter_len, strides=stride, padding='valid', input_shape=(x_train_len, 1)))
        else:
            # input layer
            if i == 0:
                model.add(Conv1D(filters=filters, kernel_size=filter_len, strides=stride, padding='valid', input_shape=(x_train_len, 1)))
            else:
                model.add(Conv1D(filters=filters, kernel_size=filter_len, strides=stride, padding='valid'))
        model.add(Activation(gParameters['activation']))
        if gParameters['pool']:
            model.add(MaxPooling1D(pool_size=pool_list[int(i / 3)]))

    model.add(Flatten())

    for layer in gParameters['dense']:
        if layer:
            model.add(Dense(layer))
            model.add(Activation(gParameters['activation']))
            if gParameters['dropout']:
                model.add(Dropout(gParameters['dropout']))
    model.add(Dense(gParameters['classes']))
    model.add(Activation(gParameters['out_activation']))

# Reference case
# model.add(Conv1D(filters=128, kernel_size=20, strides=1, padding='valid', input_shape=(P, 1)))
# model.add(Activation('relu'))
# model.add(MaxPooling1D(pool_size=1))
# model.add(Conv1D(filters=128, kernel_size=10, strides=1, padding='valid'))
# model.add(Activation('relu'))
# model.add(MaxPooling1D(pool_size=10))
# model.add(Flatten())
# model.add(Dense(200))
# model.add(Activation('relu'))
# model.add(Dropout(0.1))
# model.add(Dense(20))
# model.add(Activation('relu'))
# model.add(Dropout(0.1))
# model.add(Dense(CLASSES))
# model.add(Activation('softmax'))

    kerasDefaults = candle.keras_default_config()

    # Define optimizer
    optimizer = candle.build_optimizer(gParameters['optimizer'],
                                       gParameters['learning_rate'],
                                       kerasDefaults)

    model.summary()
    model.compile(loss=gParameters['loss'],
                  optimizer=optimizer,
                  metrics=[gParameters['metrics']])

    output_dir = gParameters['output_dir']

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # calculate trainable and non-trainable params
    gParameters.update(candle.compute_trainable_params(model))

    # set up a bunch of callbacks to do work during model training..
    model_name = gParameters['model_name']
    path = '{}/{}.autosave.model.h5'.format(output_dir, model_name)
    # checkpointer = ModelCheckpoint(filepath=path, verbose=1, save_weights_only=False, save_best_only=True)
    csv_logger = CSVLogger('{}/training.log'.format(output_dir))
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=1, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)
    candleRemoteMonitor = candle.CandleRemoteMonitor(params=gParameters)
    timeoutMonitor = candle.TerminateOnTimeOut(gParameters['timeout'])
    history = model.fit(X_train, Y_train,
                        batch_size=gParameters['batch_size'],
                        epochs=gParameters['epochs'],
                        verbose=1,
                        validation_data=(X_test, Y_test),
                        callbacks=[csv_logger, reduce_lr, candleRemoteMonitor, timeoutMonitor])

    score = model.evaluate(X_test, Y_test, verbose=0)

    if False:
        print('Test score:', score[0])
        print('Test accuracy:', score[1])
        # serialize model to JSON
        model_json = model.to_json()
        with open("{}/{}.model.json".format(output_dir, model_name), "w") as json_file:
            json_file.write(model_json)

        # serialize model to YAML
        model_yaml = model.to_yaml()
        with open("{}/{}.model.yaml".format(output_dir, model_name), "w") as yaml_file:
            yaml_file.write(model_yaml)

        # serialize weights to HDF5
        model.save_weights("{}/{}.weights.h5".format(output_dir, model_name))
        print("Saved model to disk")

        # load json and create model
        json_file = open('{}/{}.model.json'.format(output_dir, model_name), 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model_json = model_from_json(loaded_model_json)

        # load yaml and create model
        yaml_file = open('{}/{}.model.yaml'.format(output_dir, model_name), 'r')
        loaded_model_yaml = yaml_file.read()
        yaml_file.close()
        loaded_model_yaml = model_from_yaml(loaded_model_yaml)

        # load weights into new model
        loaded_model_json.load_weights('{}/{}.weights.h5'.format(output_dir, model_name))
        print("Loaded json model from disk")

        # evaluate json loaded model on test data
        loaded_model_json.compile(loss=gParameters['loss'],
                                  optimizer=gParameters['optimizer'],
                                  metrics=[gParameters['metrics']])
        score_json = loaded_model_json.evaluate(X_test, Y_test, verbose=0)

        print('json Test score:', score_json[0])
        print('json Test accuracy:', score_json[1])

        print("json %s: %.2f%%" % (loaded_model_json.metrics_names[1], score_json[1] * 100))

        # load weights into new model
        loaded_model_yaml.load_weights('{}/{}.weights.h5'.format(output_dir, model_name))
        print("Loaded yaml model from disk")

        # evaluate loaded model on test data
        loaded_model_yaml.compile(loss=gParameters['loss'],
                                  optimizer=gParameters['optimizer'],
                                  metrics=[gParameters['metrics']])
        score_yaml = loaded_model_yaml.evaluate(X_test, Y_test, verbose=0)

        print('yaml Test score:', score_yaml[0])
        print('yaml Test accuracy:', score_yaml[1])

        print("yaml %s: %.2f%%" % (loaded_model_yaml.metrics_names[1], score_yaml[1] * 100))

    return history


def main():
    gParameters = initialize_parameters()
    run(gParameters)


if __name__ == '__main__':
    main()
    try:
        K.clear_session()
    except AttributeError:      # theano does not have this function
        pass
