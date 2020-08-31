from __future__ import print_function

import numpy as np
from keras import backend as K

'''
from keras.layers import Input, Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.models import Model
from keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau

from sklearn.metrics import f1_score
'''

import keras

import p3b3 as bmk
import keras_mt_shared_cnn
import candle


def initialize_parameters(default_model='p3b3_default_model.txt'):

    # Build benchmark object
    p3b3Bmk = bmk.BenchmarkP3B3(bmk.file_path, default_model, 'keras',
                                prog='p3b3_baseline',
                                desc='Multi-task CNN for data extraction from clinical reports - Pilot 3 Benchmark 3')

    # Initialize parameters
    gParameters = candle.finalize_parameters(p3b3Bmk)
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


def run_cnn(GP, train_x, train_y, test_x, test_y,
            learning_rate=0.01,
            batch_size=10,
            epochs=10,
            dropout=0.5,
            optimizer='adam',
            wv_len=300,
            filter_sizes=[3, 4, 5],
            num_filters=[300, 300, 300],
            emb_l2=0.001,
            w_l2=0.01
            ):

    max_vocab = np.max(train_x)
    max_vocab2 = np.max(test_x)
    if max_vocab2 > max_vocab:
        max_vocab = max_vocab2

    wv_mat = np.random.randn(max_vocab + 1, wv_len).astype('float32') * 0.1

    task_list = GP['task_list']
    task_names = GP['task_names']
    num_classes = []
    for i in range(train_y.shape[1]):
        num_classes.append(np.max(train_y[:, i]) + 1)

    print('Num_classes = ', num_classes)

    kerasDefaults = candle.keras_default_config()
    optimizer_run = candle.build_optimizer(optimizer, learning_rate, kerasDefaults)

    cnn = keras_mt_shared_cnn.init_export_network(
        task_names=task_names,
        task_list=task_list,
        num_classes=num_classes,
        in_seq_len=1500,
        vocab_size=len(wv_mat),
        wv_space=wv_len,
        filter_sizes=filter_sizes,
        num_filters=num_filters,
        concat_dropout_prob=dropout,
        emb_l2=emb_l2,
        w_l2=w_l2,
        optimizer=optimizer_run)

    print(cnn.summary())

    val_labels = {}
    train_labels = []
    for i in range(train_y.shape[1]):
        if i in task_list:
            task_string = task_names[i]
            val_labels[task_string] = test_y[:, i]
            train_labels.append(np.array(train_y[:, i]))

    validation_data = ({'Input': test_x}, val_labels)

    # candleRemoteMonitor = CandleRemoteMonitor(params= GP)
    # timeoutMonitor = TerminateOnTimeOut(TIMEOUT)

    candleRemoteMonitor = candle.CandleRemoteMonitor(params=GP)
    timeoutMonitor = candle.TerminateOnTimeOut(GP['timeout'])

    history = cnn.fit(
        x=np.array(train_x),
        y=train_labels,
        batch_size=batch_size,
        epochs=epochs,
        verbose=2,
        validation_data=validation_data,
        callbacks=[candleRemoteMonitor, timeoutMonitor]
     )

    return history


def run(gParameters):

    fpath = fetch_data(gParameters)
    # Get default parameters for initialization and optimizer functions
    kerasDefaults = candle.keras_default_config()

    learning_rate = gParameters['learning_rate']
    batch_size = gParameters['batch_size']
    epochs = gParameters['epochs']
    dropout = gParameters['dropout']
    optimizer = gParameters['optimizer']
    wv_len = gParameters['wv_len']
    filter_sizes = gParameters['filter_sizes']
    filter_sets = gParameters['filter_sets']
    num_filters = gParameters['num_filters']
    emb_l2 = gParameters['emb_l2']
    w_l2 = gParameters['w_l2']

    train_x = np.load(fpath + '/train_X.npy')
    train_y = np.load(fpath + '/train_Y.npy')
    test_x = np.load(fpath + '/test_X.npy')
    test_y = np.load(fpath + '/test_Y.npy')

    for task in range(len(train_y[0, :])):
        cat = np.unique(train_y[:, task])
        train_y[:, task] = [np.where(cat == x)[0][0] for x in train_y[:, task]]
        test_y[:, task] = [np.where(cat == x)[0][0] for x in test_y[:, task]]

    run_filter_sizes = []
    run_num_filters = []

    for k in range(filter_sets):
        run_filter_sizes.append(filter_sizes + k)
        run_num_filters.append(num_filters)

    ret = run_cnn(
        gParameters,
        train_x, train_y, test_x, test_y,
        learning_rate=learning_rate,
        batch_size=batch_size,
        epochs=epochs,
        dropout=dropout,
        optimizer=optimizer,
        wv_len=wv_len,
        filter_sizes=run_filter_sizes,
        num_filters=run_num_filters,
        emb_l2=emb_l2,
        w_l2=w_l2
    )

    return ret


def main():

    gParameters = initialize_parameters()
    avg_loss = run(gParameters)
    print("Return: ", avg_loss)


if __name__ == '__main__':
    main()
    try:
        K.clear_session()
    except AttributeError:      # theano does not have this function
        pass
