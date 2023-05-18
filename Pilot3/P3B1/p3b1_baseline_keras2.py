from __future__ import print_function

import numpy as np

from tensorflow.keras import backend as K

from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model

from sklearn.metrics import f1_score

import p3b1 as bmk
import candle

from tensorflow import keras
import time


class MyCallBack(keras.callbacks.Callback):
    def __init__(self, gParameters):
        super().__init__()
        self.batchsize = gParameters['batch_size']
        self.logfreq = 10
        self.batch_begin_time = 0
        self.batch_end_time = 0
        self.max_speed = 0
        self.epoch_time = 0
        self.train_time = 0
        self.batch_log = gParameters['batch_log']

    def on_batch_begin(self, batch, logs=None):
        self.batch_begin_time = time.time()

    def on_batch_end(self, batch, logs=None):
        if batch == 0:
            return
        self.epoch_batch_count += 1
        self.train_batch_count += 1
        self.batch_time = time.time() - self.batch_begin_time
        self.epoch_time += self.batch_time

        self.batch_speed = self.batchsize / self.batch_time
        if self.batch_speed > self.max_speed:
            self.max_speed = self.batch_speed
        if self.batch_log is not None and self.batch_log is True:
            print(f"\r\nbatch {batch} time(s) {round(self.batch_time, 6)} throughput(samples/sec): {round(self.batch_speed, 3)}", flush=True)

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_batch_count = 0
        self.epoch_time = 0
        self.epoch_begin_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        self.train_time += self.epoch_time
        self.epoch_avg_speed = self.epoch_batch_count * self.batchsize / self.epoch_time
        print(f"\r\nepoch {epoch} time (s):", round(self.epoch_time, 3), " throughput(samples/sec):", round(self.epoch_avg_speed, 3), flush=True)

    def on_train_begin(self, logs=None):
        self.train_batch_count = 0
        self.train_time = 0
        self.train_begin_time = time.time()

    def on_train_end(self, logs=None):
        speed_train = (self.batchsize * self.train_batch_count) / self.train_time
        print("\r\nTotal train time(s) :" , round(self.train_time, 3), " batches:", self.train_batch_count, " batchsize:",  self.batchsize,  " throughput(samples/sec) (avg, max): ", round(speed_train, 3), round(self.max_speed, 3),flush=True)


def initialize_parameters(default_model='p3b1_default_model.txt'):

    # Build benchmark object
    p3b1Bmk = bmk.BenchmarkP3B1(bmk.file_path, default_model, 'keras',
                                prog='p3b1_baseline',
                                desc='Multi-task (DNN) for data extraction \
                                     from clinical reports - Pilot 3 Benchmark 1')

    # Initialize parameters
    gParameters = candle.finalize_parameters(p3b1Bmk)
    # bmk.logger.info('Params: {}'.format(gParameters))

    return gParameters


def fetch_data(gParameters):
    """ Downloads and decompresses the data if not locally available.
        Since the training data depends on the model definition it is not loaded,
        instead the local path where the raw data resides is returned
    """

    path = gParameters['data_url']
    fpath = candle.fetch_file(path + gParameters['train_data'], 'Pilot3', unpack=True)

    return fpath


def build_model(gParameters, kerasDefaults,
                shared_nnet_spec, individual_nnet_spec,
                input_dim, Y_train, Y_test,
                verbose=False):

    labels_train = []
    labels_test = []

    n_out_nodes = []

    for idx in range(len(Y_train)):
        truth_train = np.array(Y_train[idx], dtype='int32')
        truth_test = np.array(Y_test[idx], dtype='int32')

        mv = int(np.max(truth_train))

        label_train = np.zeros((len(truth_train), mv + 1))
        for i in range(len(truth_train)):
            label_train[i, truth_train[i]] = 1

        label_test = np.zeros((len(truth_test), mv + 1))
        for i in range(len(truth_test)):
            label_test[i, truth_test[i]] = 1

        labels_train.append(label_train)
        labels_test.append(label_test)

        n_out_nodes.append(mv + 1)

    shared_layers = []

    # input layer
    layer = Input(shape=(input_dim,), name='input')
    shared_layers.append(layer)

    # shared layers
    for k in range(len(shared_nnet_spec)):
        layer = Dense(shared_nnet_spec[k], activation=gParameters['activation'],
                      name='shared_layer_' + str(k))(shared_layers[-1])
        shared_layers.append(layer)
        if gParameters['dropout'] > 0:
            layer = Dropout(gParameters['dropout'])(shared_layers[-1])
            shared_layers.append(layer)

    # individual layers
    indiv_layers_arr = []
    models = []

    trainable_count = 0
    non_trainable_count = 0

    for idx in range(len(individual_nnet_spec)):
        indiv_layers = [shared_layers[-1]]
        for k in range(len(individual_nnet_spec[idx]) + 1):
            if k < len(individual_nnet_spec[idx]):
                layer = Dense(individual_nnet_spec[idx][k],
                              activation=gParameters['activation'],
                              name='indiv_layer_' + str(idx) + '_' + str(k))(indiv_layers[-1])
                indiv_layers.append(layer)
                if gParameters['dropout'] > 0:
                    layer = Dropout(gParameters['dropout'])(indiv_layers[-1])
                    indiv_layers.append(layer)
            else:
                layer = Dense(n_out_nodes[idx],
                              activation=gParameters['out_activation'],
                              name='out_' + str(idx))(indiv_layers[-1])
                indiv_layers.append(layer)

        indiv_layers_arr.append(indiv_layers)

        model = Model(inputs=[shared_layers[0]], outputs=[indiv_layers[-1]])

        # calculate trainable/non-trainable param count for each model
        param_counts = candle.compute_trainable_params(model)
        trainable_count += param_counts['trainable_params']
        non_trainable_count += param_counts['non_trainable_params']

        models.append(model)

    # capture total param counts
    gParameters['trainable_params'] = trainable_count
    gParameters['non_trainable_params'] = non_trainable_count
    gParameters['total_params'] = trainable_count + non_trainable_count

    # Define optimizer
    optimizer = candle.build_optimizer(gParameters['optimizer'],
                                       gParameters['learning_rate'],
                                       kerasDefaults)

    # DEBUG - verify
    if verbose:
        for k in range(len(models)):
            model = models[k]
            print('Model: ', k)
            model.summary()

    for k in range(len(models)):
        model = models[k]
        model.compile(loss=gParameters['loss'],
                      optimizer=optimizer,
                      metrics=[gParameters['metrics']])

    return models, labels_train, labels_test


def train_model(gParameters, models,
                X_train, Y_train,
                X_test, Y_test,
                fold, verbose=False):

    base_run_id = gParameters['run_id']
    my_hook = MyCallBack(gParameters)

    for epoch in range(gParameters['epochs']):
        for k in range(len(models)):

            model = models[k]

            gParameters['run_id'] = base_run_id + ".{}.{}.{}".format(fold, epoch, k)
            candleRemoteMonitor = candle.CandleRemoteMonitor(params=gParameters)
            timeoutMonitor = candle.TerminateOnTimeOut(gParameters['timeout'])

            model.fit({'input': X_train[k]}, {'out_' + str(k): Y_train[k]},
                      epochs=1, verbose=verbose,
                      callbacks=[candleRemoteMonitor, timeoutMonitor, my_hook],
                      batch_size=gParameters['batch_size'],
                      validation_data=(X_test[k], Y_test[k]))

    return models


def evaluate_model(X_test, truths_test, labels_test, models):

    # retrieve truth-pred pair
    avg_loss = 0.0
    ret = []

    for k in range(len(models)):
        ret_k = []

        feature_test = X_test[k]
        truth_test = truths_test[k]
        label_test = labels_test[k]
        model = models[k]

        loss = model.evaluate(feature_test, label_test)
        avg_loss = avg_loss + loss[0]
        print("In EVALUATE loss: ", loss)

        pred = model.predict(feature_test)

        ret_k.append(truth_test)
        ret_k.append(np.argmax(pred, axis=1))

        ret.append(ret_k)

    avg_loss = avg_loss / float(len(models))
    ret.append(avg_loss)

    return ret


def run(gParameters):

    fpath = fetch_data(gParameters)
    # Get default parameters for initialization and optimizer functions
    kerasDefaults = candle.keras_default_config()

    # Construct structures common to all folds
#    shared_nnet_spec = []
#    elem = gParameters['shared_nnet_spec'].split(',')
#    for el in elem:
#        shared_nnet_spec.append(int(el))

#    individual_nnet_spec = []
#    indiv = gParameters['ind_nnet_spec'].split(':')
#    for ind in indiv:
#        indiv_nnet_spec = []
#        elem = ind.split(',')
#        for el in elem:
#            indiv_nnet_spec.append(int(el))
#        individual_nnet_spec.append(indiv_nnet_spec)

    shared_nnet_spec = gParameters['shared_nnet_spec']
    individual_nnet_spec = gParameters['ind_nnet_spec']

    # Construct features common to all folds
    features = []
    feat = gParameters['feature_names'].split(':')
    for f in feat:
        features.append(f)

    n_feat = len(feat)
    print('Feature names:')
    for i in range(n_feat):
        print(features[i])

    # initialize arrays for all the features
    truth_array = [[] for _ in range(n_feat)]
    pred_array = [[] for _ in range(n_feat)]
    avg_loss = 0.0

    # stdout display level
    verbose = True

    # per fold
    for fold in range(gParameters['n_fold']):

        # build data
        X_train, Y_train, X_test, Y_test = bmk.build_data(len(individual_nnet_spec), fold, fpath)

        # build model
        input_dim = len(X_train[0][0])
        models, labels_train, labels_test = build_model(gParameters, kerasDefaults,
                                                        shared_nnet_spec, individual_nnet_spec,
                                                        input_dim, Y_train, Y_test, verbose)

        # train model
        models = train_model(gParameters, models,
                             X_train, labels_train,
                             X_test, labels_test,
                             fold, verbose)

        # evaluate model
        ret = evaluate_model(X_test, Y_test, labels_test, models)

        for i in range(n_feat):
            truth_array[i].extend(ret[i][0])
            pred_array[i].extend(ret[i][1])

        avg_loss += ret[-1]

    avg_loss /= float(gParameters['n_fold'])

    for task in range(n_feat):
        print('Task', task + 1, ':', features[task], '- Macro F1 score',
              f1_score(truth_array[task], pred_array[task], average='macro'))
        print('Task', task + 1, ':', features[task], '- Micro F1 score',
              f1_score(truth_array[task], pred_array[task], average='micro'))

    return avg_loss


def main():

    gParameters = initialize_parameters()
    avg_loss = run(gParameters)
    print("Average loss: ", avg_loss)


if __name__ == '__main__':
    main()
    try:
        K.clear_session()
    except AttributeError:      # theano does not have this function
        pass
