import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
import numpy as np
import os

import pickle

import sys

import p3b2 as bmk
import candle


def initialize_parameters(default_model='p3b2_default_model.txt'):

    # Build benchmark object
    p3b2Bmk = bmk.BenchmarkP3B2(bmk.file_path, default_model, 'keras',
                                prog='p3b2_baseline',
                                desc='Multi-task (DNN) for data extraction from \
                                clinical reports - Pilot 3 Benchmark 1')

    # Initialize parameters
    gParameters = candle.finalize_parameters(p3b2Bmk)
    #bmk.logger.info('Params: {}'.format(gParameters))

    return gParameters


class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def run(gParameters):

    origin = gParameters['data_url']
    train_data = gParameters['train_data']
    data_loc = candle.fetch_file(origin+train_data, untar=True, md5_hash=None, subdir='Pilot3')

    print('Data downloaded and stored at: ' + data_loc)
    data_path = os.path.dirname(data_loc)
    print(data_path)

    kerasDefaults = candle.keras_default_config()

    rnn_size = gParameters['rnn_size']
    n_layers = gParameters['n_layers']
    learning_rate = gParameters['learning_rate']
    dropout = gParameters['dropout']
    recurrent_dropout = gParameters['recurrent_dropout']
    n_epochs = gParameters['epochs']
    data_train = data_path+'/data.pkl'
    verbose = gParameters['verbose']
    savedir = gParameters['output_dir']
    do_sample = gParameters['do_sample']
    temperature = gParameters['temperature']
    primetext = gParameters['primetext']
    length = gParameters['length']

    # load data from pickle
    f = open(data_train, 'rb')

    if (sys.version_info > (3, 0)):
        classes = pickle.load(f, encoding='latin1')
        chars = pickle.load(f, encoding='latin1')
        char_indices = pickle.load(f, encoding='latin1')
        indices_char = pickle.load(f, encoding='latin1')

        maxlen = pickle.load(f, encoding='latin1')
        step = pickle.load(f, encoding='latin1')

        X_ind = pickle.load(f, encoding='latin1')
        y_ind = pickle.load(f, encoding='latin1')
    else:
        classes = pickle.load(f)
        chars = pickle.load(f)
        char_indices = pickle.load(f)
        indices_char = pickle.load(f)

        maxlen = pickle.load(f)
        step = pickle.load(f)

        X_ind = pickle.load(f)
        y_ind = pickle.load(f)

    f.close()

    [s1, s2] = X_ind.shape
    print(X_ind.shape)
    print(y_ind.shape)
    print(maxlen)
    print(len(chars))

    X = np.zeros((s1, s2, len(chars)), dtype=np.bool)
    y = np.zeros((s1, len(chars)), dtype=np.bool)

    for i in range(s1):
        for t in range(s2):
            X[i, t, X_ind[i, t]] = 1
        y[i, y_ind[i]] = 1

    # build the model: a single LSTM
    if verbose:
        print('Build model...')

    model = Sequential()

    # for rnn_size in rnn_sizes:
    for k in range(n_layers):
        if k < n_layers - 1:
            ret_seq = True
        else:
            ret_seq = False

        if k == 0:
            model.add(LSTM(rnn_size, input_shape=(maxlen, len(chars)), return_sequences=ret_seq,
                           dropout=dropout, recurrent_dropout=recurrent_dropout))
        else:
            model.add(LSTM(rnn_size, dropout=dropout, recurrent_dropout=recurrent_dropout,
                      return_sequences=ret_seq))

    model.add(Dense(len(chars)))
    model.add(Activation(gParameters['activation']))

    optimizer = candle.build_optimizer(gParameters['optimizer'],
                                       gParameters['learning_rate'],
                                       kerasDefaults)

    model.compile(loss=gParameters['loss'], optimizer=optimizer)

    if verbose:
        model.summary()

    for iteration in range(1, n_epochs + 1):
        if verbose:
            print()
            print('-' * 50)
            print('Iteration', iteration)

        history = LossHistory()
        model.fit(X, y, batch_size=100, epochs=1, callbacks=[history])

        loss = history.losses[-1]
        if verbose:
            print(loss)

        dirname = savedir
        if len(dirname) > 0 and not dirname.endswith('/'):
            dirname = dirname + '/'

        if not os.path.exists(dirname):
            os.makedirs(dirname)

        # serialize model to JSON
        model_json = model.to_json()
        with open(dirname + "/model_" + str(iteration) + "_" + "{:f}".format(loss) + ".json", "w") as json_file:
            json_file.write(model_json)

        # serialize weights to HDF5
        model.save_weights(dirname + "/model_" + str(iteration) + "_" + "{:f}".format(loss) + ".h5")

        if verbose:
            print("Checkpoint saved.")

        if do_sample:
            outtext = open(dirname + "/example_" + str(iteration) + "_" + "{:f}".format(loss) + ".txt", "w", encoding='utf-8')

            diversity = temperature

            outtext.write('----- diversity:' + str(diversity) + "\n")

            generated = ''
            seedstr = primetext

            outtext.write('----- Generating with seed: "' + seedstr + '"' + "\n")

            sentence = " " * maxlen

            # class_index = 0
            generated += sentence
            outtext.write(generated)

            for c in seedstr:
                sentence = sentence[1:] + c
                x = np.zeros((1, maxlen, len(chars)))
                for t, char in enumerate(sentence):
                    x[0, t, char_indices[char]] = 1.

                preds = model.predict(x, verbose=verbose)[0]
                next_index = sample(preds, diversity)
                next_char = indices_char[next_index]

                generated += c

                outtext.write(c)

            for i in range(length):
                x = np.zeros((1, maxlen, len(chars)))
                for t, char in enumerate(sentence):
                    x[0, t, char_indices[char]] = 1.

                preds = model.predict(x, verbose=verbose)[0]
                next_index = sample(preds, diversity)
                next_char = indices_char[next_index]

                generated += next_char
                sentence = sentence[1:] + next_char

            if (sys.version_info > (3, 0)):
                outtext.write(generated + '\n')
            else:
                outtext.write(generated.decode('utf-8').encode('utf-8') + '\n')

            outtext.close()


if __name__ == "__main__":

    gParameters = initialize_parameters()
    run(gParameters)
