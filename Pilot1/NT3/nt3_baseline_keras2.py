from __future__ import print_function

import numpy as np

import argparse

from keras import backend as K

from keras.layers import Input, Dense, Dropout, Activation, Conv1D, MaxPooling1D, Flatten
from keras.optimizers import SGD, Adam, RMSprop
from keras.models import Sequential, Model, model_from_json, model_from_yaml
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau

import nt3 as benchmark
import default_utils
import keras_utils
import data_utils

from solr_keras import CandleRemoteMonitor, compute_trainable_params, TerminateOnTimeOut


def initialize_parameters():

    # Build benchmark object
    nt3Bmk = benchmark.BenchmarkNT3(benchmark.file_path, 'nt3_default_model.txt', 'keras',
    prog='nt3_baseline', desc='Train Autoencoder - Pilot 1 Benchmark NT3')
    
    # Initialize parameters
    gParameters = default_utils.initialize_parameters(nt3Bmk)
    csv_logger = CSVLogger('{}/params.log'.format(gParameters))
    #benchmark.logger.info('Params: {}'.format(gParameters))

    return gParameters


def load_data(gParameters):

    path = gParameters['data_url']
    train_file = default_utils.fetch_file(path + gParameters['train_data'], 'Pilot1')
    test_file = default_utils.fetch_file(path + gParameters['test_data'], 'Pilot1')
    
    X_train, Y_train, X_test, Y_test = benchmark.load_data(train_file, test_file, gParameters)

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
    
    data = (X_train, Y_train, X_test, Y_test, x_train_len)
    
    return data


def run(gParameters, data):

    # Get default parameters for initialization and optimizer functions
    kerasDefaults = default_utils.keras_default_config()

    # 'unfold' data
    X_train, Y_train, X_test, Y_test, x_train_len = data

    seed = gParameters['rng_seed']

    # Initialize weights and learning rule
    initializer_weights = keras_utils.build_initializer(gParameters['initialization'], kerasDefaults, seed)
    initializer_bias = keras_utils.build_initializer('constant', kerasDefaults, 0.)
    
    # Define model architecture
    model = Sequential()

    layer_list = list(range(0, len(gParameters['conv']), 3))
    for l, i in enumerate(layer_list):
        filters = gParameters['conv'][i]
        filter_len = gParameters['conv'][i+1]
        stride = gParameters['conv'][i+2]
        print(int(i/3), filters, filter_len, stride)
        if gParameters['pool']:
            pool_list=gParameters['pool']
            if type(pool_list) != list:
                pool_list=list(pool_list)

        if filters <= 0 or filter_len <= 0 or stride <= 0:
                break
        if 'locally_connected' in gParameters:
                model.add(LocallyConnected1D(filters, filter_len, strides=stride, padding='valid', input_shape=(x_train_len, 1)))
        else:
            #input layer
            if i == 0:
                model.add(Conv1D(filters=filters, kernel_size=filter_len, strides=stride, padding='valid', input_shape=(x_train_len, 1)))
            else:
                model.add(Conv1D(filters=filters, kernel_size=filter_len, strides=stride, padding='valid'))
        model.add(Activation(gParameters['activation']))
        if gParameters['pool']:
                model.add(MaxPooling1D(pool_size=pool_list[int(i/3)]))

    model.add(Flatten())

    for layer in gParameters['dense']:
        if layer:
            model.add(Dense(layer))
            model.add(Activation(gParameters['activation']))
            if gParameters['drop']:
                    model.add(Dropout(gParameters['drop']))
    model.add(Dense(gParameters['classes']))
    model.add(Activation(gParameters['out_activation']))


    # Define optimizer
    optimizer = keras_utils.build_optimizer(gParameters['optimizer'],
                                            gParameters['learning_rate'],
                                            kerasDefaults)

    # Compile and display model
    model.compile(loss=gParameters['loss'], optimizer=optimizer, metrics=[gParameters['metrics']])
    model.summary()

    # calculate trainable and non-trainable params
    gParameters.update(compute_trainable_params(model))

    # set up a bunch of callbacks to do work during model training..
    model_name = gParameters['model_name']
    output_dir = gParameters['output_dir']
    path = '{}/{}.autosave.model.h5'.format(output_dir, model_name)
    # checkpointer = ModelCheckpoint(filepath=path, verbose=1, save_weights_only=False, save_best_only=True)
    csv_logger = CSVLogger('{}/training.log'.format(output_dir))
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=1, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)
    candleRemoteMonitor = CandleRemoteMonitor(params=gParameters)
    timeoutMonitor = TerminateOnTimeOut(gParameters['timeout'])
    history = model.fit(X_train, Y_train,
                    batch_size=gParameters['batch_size'],
                    epochs=gParameters['epochs'],
                    verbose=1,
                    validation_data=(X_test, Y_test),
                    callbacks = [csv_logger, reduce_lr, candleRemoteMonitor, timeoutMonitor])

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
        model.save_weights("{}/{}.model.h5".format(output_dir, model_name))
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
        loaded_model_json.load_weights('{}/{}.model.h5'.format(output_dir, model_name))
        print("Loaded json model from disk")

        # evaluate json loaded model on test data
        loaded_model_json.compile(loss=gParameters['loss'],
            optimizer=gParameters['optimizer'],
            metrics=[gParameters['metrics']])
        score_json = loaded_model_json.evaluate(X_test, Y_test, verbose=0)

        print('json Test score:', score_json[0])
        print('json Test accuracy:', score_json[1])

        print("json %s: %.2f%%" % (loaded_model_json.metrics_names[1], score_json[1]*100))

        # load weights into new model
        loaded_model_yaml.load_weights('{}/{}.model.h5'.format(output_dir, model_name))
        print("Loaded yaml model from disk")

        # evaluate loaded model on test data
        loaded_model_yaml.compile(loss=gParameters['loss'],
            optimizer=gParameters['optimizer'],
            metrics=[gParameters['metrics']])
        score_yaml = loaded_model_yaml.evaluate(X_test, Y_test, verbose=0)

        print('yaml Test score:', score_yaml[0])
        print('yaml Test accuracy:', score_yaml[1])

        print("yaml %s: %.2f%%" % (loaded_model_yaml.metrics_names[1], score_yaml[1]*100))

    return history



def main():

    gParameters = initialize_parameters()
    data = load_data(gParameters)
    run(gParameters, data)



if __name__ == '__main__':
    main()
    try:
        K.clear_session()
    except AttributeError:      # theano does not have this function
        pass
