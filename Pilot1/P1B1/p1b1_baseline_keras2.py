from __future__ import print_function

import argparse
import h5py

import numpy as np

import keras
from keras import backend as K
from keras import optimizers
from keras.models import Model, Sequential
from keras.layers import Activation, Dense, Dropout, Input, Lambda
from keras.initializers import RandomUniform
from keras.callbacks import Callback, ModelCheckpoint, ReduceLROnPlateau, ProgbarLogger
from keras.metrics import binary_crossentropy
from scipy.stats.stats import pearsonr

import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    from sklearn.metrics import r2_score

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import p1b1
import p1_common
import p1_common_keras
from solr_keras import CandleRemoteMonitor


def get_p1b1_parser():
	parser = argparse.ArgumentParser(prog='p1b1_baseline', formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description='Train Autoencoder - Pilot 1 Benchmark 1')
	return p1b1.common_parser(parser)


def covariance(x, y):
    return K.mean(x * y) - K.mean(x) * K.mean(y)


def corr(y_true, y_pred):
    cov = covariance(y_true, y_pred)
    var1 = covariance(y_true, y_true)
    var2 = covariance(y_pred, y_pred)
    return cov / (K.sqrt(var1 * var2) + K.epsilon())


def xent(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred)


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


def initialize_parameters():
    # Get command-line parameters
    parser = get_p1b1_parser()
    args = parser.parse_args()
    #print('Args:', args)
    # Get parameters from configuration file
    fileParameters = p1b1.read_config_file(args.config_file)
    #print ('Params:', fileParameters)
    # Consolidate parameter set. Command-line parameters overwrite file configuration
    gParameters = p1_common.args_overwrite_config(args, fileParameters)
    print(gParameters)
    return gParameters


def run(gParameters):
    # Construct extension to save model
    ext = p1b1.extension_from_parameters(gParameters, '.keras')
    logfile =  gParameters['logfile'] if gParameters['logfile'] else gParameters['save']+ext+'.log'
    p1b1.logger.info('Params: {}'.format(gParameters))

    # Get default parameters for initialization and optimizer functions
    kerasDefaults = p1_common.keras_default_config()
    seed = gParameters['rng_seed']

    # Load dataset
    # X_train, X_val, X_test = p1b1.load_data(gParameters, seed)
    # with h5py.File('x_cache.h5', 'w') as hf:
        # hf.create_dataset("train",  data=X_train)
        # hf.create_dataset("val",  data=X_val)
        # hf.create_dataset("test",  data=X_test)

    with h5py.File('x_cache.h5', 'r') as hf:
        X_train = hf['train'][:]
        X_val = hf['val'][:]
        X_test = hf['test'][:]

    print("Shape X_train: ", X_train.shape)
    print("Shape X_val: ", X_val.shape)
    print("Shape X_test: ", X_test.shape)

    print("Range X_train --> Min: ", np.min(X_train), ", max: ", np.max(X_train))
    print("Range X_val --> Min: ", np.min(X_val), ", max: ", np.max(X_val))
    print("Range X_test --> Min: ", np.min(X_test), ", max: ", np.max(X_test))

    # Initialize weights and learning rule
    initializer_weights = p1_common_keras.build_initializer(gParameters['initialization'], kerasDefaults, seed)
    initializer_bias = p1_common_keras.build_initializer('constant', kerasDefaults, 0.)

    input_dim = X_train.shape[1]
    output_dim = input_dim
    latent_dim = gParameters['latent_dim']

    vae = gParameters['vae']
    activation = gParameters['activation']
    dropout = gParameters['drop']
    layers = gParameters['dense']

    if layers != None:
        if type(layers) != list:
            layers = list(layers)
    else:
        layers = []

    # Encoder Part
    input_vector = Input(shape=(input_dim,))
    h = input_vector
    for i, l in enumerate(layers):
        if l > 0:
            h = Dense(l, activation=activation,
                      kernel_initializer=initializer_weights,
                      bias_initializer=initializer_bias)(h)
            if dropout > 0:
                h = Dropout(dropout)(h)

    if not vae:
        encoded = Dense(latent_dim, activation=activation,
                        kernel_initializer=initializer_weights,
                        bias_initializer=initializer_bias)(h)
    else:
        epsilon_std = 1.0
        z_mean = Dense(latent_dim, name='z_mean')(h)
        z_log_var = Dense(latent_dim, name='z_log_var')(h)
        encoded = z_mean

        def vae_loss(x, x_decoded_mean):
            xent_loss = binary_crossentropy(x, x_decoded_mean)
            kl_loss = - 0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
            # return xent_loss
            return xent_loss + kl_loss

        def sampling(params):
            z_mean_, z_log_var_ = params
            batch_size = K.shape(z_mean_)[0]
            epsilon = K.random_normal(shape=(batch_size, latent_dim),
                                      mean=0., stddev=epsilon_std)
            return z_mean_ + K.exp(z_log_var_ / 2) * epsilon

        z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

    encoder = Model(input_vector, encoded)

    # Decoder Part
    decoder_input = Input(shape=(latent_dim,))
    h = decoder_input
    for i, l in reversed(list(enumerate(layers))):
        if l > 0:
            if dropout > 0:
                h = Dropout(dropout)(h)
            h = Dense(l, activation=activation,
                      kernel_initializer=initializer_weights,
                      bias_initializer=initializer_bias)(h)

    decoded = Dense(input_dim, kernel_initializer=initializer_weights,
                    bias_initializer=initializer_bias)(h)

    decoder = Model(decoder_input, decoded)

    # Build and compile autoencoder model
    if not vae:
        model = Model(input_vector, decoder(encoded))
        loss = gParameters['loss']
        metrics = [xent, corr]
    else:
        model = Model(input_vector, decoder(z))
        loss = vae_loss
        metrics = [xent, corr]

    # Define optimizer
    optimizer = p1_common_keras.build_optimizer(gParameters['optimizer'],
                                                gParameters['learning_rate'],
                                                kerasDefaults)

    p1b1.logger.debug('Model: {}'.format(model.to_json()))

    # Compile and display model
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    model.summary()

    # calculate trainable and non-trainable params
    trainable_count = int(
        np.sum([K.count_params(p) for p in set(model.trainable_weights)]))
    non_trainable_count = int(
        np.sum([K.count_params(p) for p in set(model.non_trainable_weights)]))
    gParameters['trainable_params'] = trainable_count
    gParameters['non_trainable_params'] = non_trainable_count
    gParameters['total_params'] = trainable_count + non_trainable_count

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.0001)
    candleRemoteMonitor = CandleRemoteMonitor(params=gParameters)

    # Seed random generator for training
    np.random.seed(seed)

    X_val2 = np.copy(X_val)
    np.random.shuffle(X_val2)
    start_scores = p1b1.evaluate_autoencoder(X_val, X_val2)
    print('\nBetween random permutations of validation data:', start_scores)

    history = model.fit(X_train, X_train,
                     batch_size=gParameters['batch_size'],
                     epochs=gParameters['epochs'],
                     # callbacks=[candleRemoteMonitor],
                     callbacks=[reduce_lr, candleRemoteMonitor],
                     validation_data=(X_val, X_val))

    # model save
    #save_filepath = "model_ae_W_" + ext
    #model.save_weights(save_filepath)

    # Evalute model on test set
    X_pred = model.predict(X_test)
    scores = p1b1.evaluate_autoencoder(X_pred, X_test)
    print('\nEvaluation on test data:', scores)

    # diff = X_pred - X_test
    # plt.hist(diff.ravel(), bins='auto')
    # plt.title("Histogram of Errors with 'auto' bins")
    # plt.savefig('histogram_keras.png')

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
