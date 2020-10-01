from __future__ import print_function

import numpy as np

from keras import backend as K

from keras.layers import Input, Dense, Dropout
from keras.models import Model

from sklearn.metrics import f1_score

import p3b1 as bmk
import candle


def initialize_parameters(default_model='p3b1_default_model.txt'):

    # Build benchmark object
    p3b1Bmk = bmk.BenchmarkP3B1(bmk.file_path, default_model, 'keras',
                                prog='p3b1_baseline',
                                desc='Multi-task (DNN) for data extraction \
                                     from clinical reports - Pilot 3 Benchmark 1')

    # Initialize parameters
    gParameters = candle.finalize_parameters(p3b1Bmk)
    #bmk.logger.info('Params: {}'.format(gParameters))

    return gParameters


def fetch_data(gParameters):
    """ Downloads and decompresses the data if not locally available.
        Since the training data depends on the model definition it is not loaded,
        instead the local path where the raw data resides is returned
    """

    path = gParameters['data_url']
    fpath = candle.fetch_file(path + gParameters['train_data'], 'Pilot3', untar=True)

    return fpath


def build_model(gParameters, kerasDefaults,
                shared_nnet_spec, individual_nnet_spec,
                input_dim, Y_train, Y_test,
                verbose=False):

    labels_train = []
    labels_test = []

    n_out_nodes = []

    for l in range(len(Y_train)):
        truth_train = np.array(Y_train[l], dtype='int32')
        truth_test = np.array(Y_test[l], dtype='int32')

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

    for l in range(len(individual_nnet_spec)):
        indiv_layers = [shared_layers[-1]]
        for k in range(len(individual_nnet_spec[l]) + 1):
            if k < len(individual_nnet_spec[l]):
                layer = Dense(individual_nnet_spec[l][k],
                              activation=gParameters['activation'],
                              name='indiv_layer_' + str(l) + '_' + str(k))(indiv_layers[-1])
                indiv_layers.append(layer)
                if gParameters['dropout'] > 0:
                    layer = Dropout(gParameters['dropout'])(indiv_layers[-1])
                    indiv_layers.append(layer)
            else:
                layer = Dense(n_out_nodes[l],
                              activation=gParameters['out_activation'],
                              name='out_' + str(l))(indiv_layers[-1])
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

    for epoch in range(gParameters['epochs']):
        for k in range(len(models)):

            model = models[k]

            gParameters['run_id'] = base_run_id + ".{}.{}.{}".format(fold, epoch, k)
            candleRemoteMonitor = candle.CandleRemoteMonitor(params=gParameters)
            timeoutMonitor = candle.TerminateOnTimeOut(gParameters['timeout'])

            model.fit({'input': X_train[k]}, {'out_' + str(k): Y_train[k]},
                      epochs=1, verbose=verbose,
                      callbacks=[candleRemoteMonitor, timeoutMonitor],
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
        print('Task', task+1, ':', features[task], '- Macro F1 score',
              f1_score(truth_array[task], pred_array[task], average='macro'))
        print('Task', task+1, ':', features[task], '- Micro F1 score',
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
