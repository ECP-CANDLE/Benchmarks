from __future__ import print_function
import pandas as pd
import numpy as np
import os
import tensorflow
from tensorflow.keras import backend as K
os.environ["CUDA_VISIBLE_DEVICES"]="2"
from tensorflow.keras.layers import Dense, Dropout, Activation, Conv1D, MaxPooling1D, Flatten, LocallyConnected1D
from tensorflow.keras.models import Sequential, model_from_json, model_from_yaml
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import CSVLogger, ReduceLROnPlateau

from sklearn.preprocessing import MaxAbsScaler
from abstain_functions import abs_definitions

import nt3 as bmk
import candle
import pickle
additional_definitions = abs_definitions

required = bmk.required


class BenchmarkNT3Abs(candle.Benchmark):
    def set_locals(self):
        """Functionality to set variables specific for the benchmark
        - required: set of required parameters for the benchmark.
        - additional_definitions: list of dictionaries describing the additional parameters for the
        benchmark.
        """

        if required is not None:
            self.required = set(bmk.required)
        if additional_definitions is not None:
            self.additional_definitions = abs_definitions + bmk.additional_definitions

                            
def initialize_parameters(default_model='nt3_noise_model.txt'):

    # Build benchmark object
    nt3Bmk = BenchmarkNT3Abs(
        bmk.file_path,
        default_model,
        'keras',
        prog='nt3_abstention',
        desc='1D CNN to classify RNA sequence data in normal or tumor classes')

    # Initialize parameters
    gParameters = candle.finalize_parameters(nt3Bmk)

    return gParameters

    
def load_data(path, gParameters):

    # Rewrite this function to handle pickle files instead
    print("Loading data...")
    data = pickle.load(open(path, 'rb'))
    X=data[0]
    y=data[1]
    polluted_inds = data[2]
    cluster_inds = data[3]
    size = X.shape[0]
    X_train = X[0:(int)(0.8*size)]
    X_test = X[(int)(0.8*size):]
    Y_train = y[0:(int)(0.8*size)]
    Y_test = y[(int)(0.8*size):]
    #df_train = (pd.read_csv(train_path, header=None).values).astype('float32')
    #df_test = (pd.read_csv(test_path, header=None).values).astype('float32')
    #X_train,Y_train, X_test, Y_test = data
    #polluted_inds = []
    #cluster_inds=[]
    print('done')

    
    print('df_train shape:', X_train.shape)
    print('df_test shape:', X_test.shape)

    return X_train, Y_train, X_test, Y_test, polluted_inds, cluster_inds


def run(gParameters):

    print('Params:', gParameters)

    data_file = gParameters['cf_noise']
   # file_test = gParameters['test_data']
    url = gParameters['data_url']

    #train_file = candle.get_file(file_train, url + file_train, cache_subdir='Pilot1')
    #test_file = candle.get_file(file_test, url + file_test, cache_subdir='Pilot1')

    X_train, Y_train, X_test, Y_test, polluted_inds, cluster_inds = load_data(data_file, gParameters)

    # add extra class for abstention
    # first reverse the to_categorical
    Y_train = np.argmax(Y_train, axis=1)
    Y_test = np.argmax(Y_test, axis=1)
    Y_train, Y_test = candle.modify_labels(gParameters['classes'] + 1, Y_train, Y_test)
    # print(Y_test)

    print('X_train shape:', X_train.shape)
    print('X_test shape:', X_test.shape)

    print('Y_train shape:', Y_train.shape)
    print('Y_test shape:', Y_test.shape)

    x_train_len = X_train.shape[1]

    # this reshaping is critical for the Conv1D to work

    #X_train = np.expand_dims(X_train, axis=2)
    #X_test = np.expand_dims(X_test, axis=2)

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

    # modify the model for abstention
    model = candle.add_model_output(model, mode='abstain', num_add=1, activation=gParameters['out_activation'])

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

    # Configure abstention model
    nb_classes = gParameters['classes']
    mask = np.zeros(nb_classes + 1)
    mask[nb_classes] = 1.0
    print("Mask is ", mask)
    alpha0 = gParameters['alpha']
    if isinstance(gParameters['max_abs'], list):
        max_abs = gParameters['max_abs'][0]
    else:
        max_abs = gParameters['max_abs']

    print("Initializing abstention callback with: \n")
    print("alpha0 ", alpha0)
    print("alpha_scale_factor ", gParameters['alpha_scale_factor'])
    print("min_abs_acc ", gParameters['min_acc'])
    print("max_abs_frac ", max_abs)
    print("acc_gain ", gParameters['acc_gain'])
    print("abs_gain ", gParameters['abs_gain'])

    abstention_cbk = candle.AbstentionAdapt_Callback(acc_monitor='val_abstention_acc',
                                                     abs_monitor='val_abstention',
                                                     init_abs_epoch=gParameters['init_abs_epoch'],
                                                     alpha0=alpha0,
                                                     alpha_scale_factor=gParameters['alpha_scale_factor'],
                                                     min_abs_acc=gParameters['min_acc'],
                                                     max_abs_frac=max_abs,
                                                     acc_gain=gParameters['acc_gain'],
                                                     abs_gain=gParameters['abs_gain'])

    model.compile(loss=candle.abstention_loss(abstention_cbk.alpha, mask),
                  optimizer=optimizer,
                  metrics=[candle.abstention_acc_metric(nb_classes),
                           # candle.acc_class_i_metric(1),
                           # candle.abstention_acc_class_i_metric(nb_classes, 1),
                           candle.abstention_metric(nb_classes)])

    # model.compile(loss=abs_loss,
    #              optimizer=optimizer,
    #              metrics=abs_acc)

    # model.compile(loss=gParameters['loss'],
    #              optimizer=optimizer,
    #              metrics=[gParameters['metrics']])

    output_dir = gParameters['output_dir']

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # calculate trainable and non-trainable params
    gParameters.update(candle.compute_trainable_params(model))

    # set up a bunch of callbacks to do work during model training..
    model_name = gParameters['model_name']
    # path = '{}/{}.autosave.model.h5'.format(output_dir, model_name)
    # checkpointer = ModelCheckpoint(filepath=path, verbose=1, save_weights_only=False, save_best_only=True)
    print(output_dir)
    csv_logger = CSVLogger("{}/training.log".format(output_dir))
    reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                  factor=0.1, patience=10, verbose=1, mode='auto',
                                  epsilon=0.0001, cooldown=0, min_lr=0)

    candleRemoteMonitor = candle.CandleRemoteMonitor(params=gParameters)
    timeoutMonitor = candle.TerminateOnTimeOut(gParameters['timeout'])

    # n_iters = 1

    # val_labels = {"activation_5": Y_test}
    # for epoch in range(gParameters['epochs']):
    #    print('Iteration = ', epoch)
    history = model.fit(X_train, Y_train,
                        batch_size=gParameters['batch_size'],
                        epochs=gParameters['epochs'],
                        # initial_epoch=epoch,
                        # epochs=epoch + n_iters,
                        verbose=1,
                        validation_data=(X_test, Y_test),
                        # callbacks=[csv_logger, reduce_lr, candleRemoteMonitor, timeoutMonitor]) # , abstention_cbk])
                        callbacks=[csv_logger, reduce_lr, candleRemoteMonitor, timeoutMonitor, abstention_cbk])

    #    ret, alpha = adjust_alpha(gParameters, X_test, Y_test, val_labels, model, alpha, [nb_classes+1])

    score = model.evaluate(X_test, Y_test, verbose=0)

    if len(polluted_inds) > 0:
        y_pred = model.predict(X_test)
        abstain_inds = []
        for i in range(y_pred.shape[0]):
            if np.argmax(y_pred[i]) == nb_classes:
                abstain_inds.append(i)

        # Cluster indices and polluted indices are wrt to entire train + test dataset
        # whereas y_pred only contains test dataset so add offset for correct indexing
        offset_testset = Y_train.shape[0]
        abstain_inds=[i+offset_testset for i in abstain_inds]
    
        polluted_percentage = c = np.sum([el in polluted_inds for el in abstain_inds])/np.max([len(abstain_inds),1])
        print("Percentage of abstained samples that were polluted {}".format(polluted_percentage))

        cluster_inds_test = list(filter(lambda cluster_inds: cluster_inds >= offset_testset, cluster_inds))
        cluster_inds_test_abstain = [el in abstain_inds for el in cluster_inds_test]
        cluster_percentage = c = np.sum(cluster_inds_test_abstain)/len(cluster_inds_test)
        print("Percentage of cluster (in test set) that was abstained {}".format(cluster_percentage))

        unabstain_inds = []
        for i in range(y_pred.shape[0]):
            if np.argmax(y_pred[i]) != nb_classes and (i+offset_testset in cluster_inds_test):
                unabstain_inds.append(i)
        # Make sure number of unabstained indices in cluster test set plus number of abstainsed indices in cluster test set
        # equals number of indices in cluster in the test set
        assert(len(unabstain_inds)+np.sum(cluster_inds_test_abstain) == len(cluster_inds_test))
        score_cluster = 1 if len(unabstain_inds)==0 else model.evaluate(X_test[unabstain_inds], Y_test[unabstain_inds])[1]
        print("Accuracy of unabastained cluster {}".format(score_cluster))
    
        pickle.dump({'Abs polluted': polluted_percentage, 'Abs val cluster': cluster_percentage, 'Abs val acc': score_cluster}, open("{}/cluster_trace.pkl".format(output_dir), "wb"))

    alpha_trace = open(output_dir + "/alpha_trace", "w+")
    for alpha in abstention_cbk.alphavalues:
        alpha_trace.write(str(alpha) + '\n')
    alpha_trace.close()

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
