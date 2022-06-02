from __future__ import print_function

import numpy as np
import h5py

import tensorflow.keras as keras
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, Input, Lambda, AlphaDropout
from tensorflow.keras.callbacks import Callback, ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler, TensorBoard
from tensorflow.keras.metrics import binary_crossentropy, mean_squared_error
from scipy.stats.stats import pearsonr
from sklearn.manifold import TSNE

import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    from sklearn.metrics import r2_score
    from sklearn.metrics import accuracy_score

import matplotlib as mpl
mpl.use('Agg')

import p1b1
import candle

np.set_printoptions(precision=4)


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


def initialize_parameters(default_model='p1b1_default_model.txt'):

    # Build benchmark object
    p1b1Bmk = p1b1.BenchmarkP1B1(p1b1.file_path, default_model, 'keras',
                                 prog='p1b1_baseline',
                                 desc='Multi-task (DNN) for data extraction from clinical reports - Pilot 3 Benchmark 1')

    # Initialize parameters
    gParameters = candle.finalize_parameters(p1b1Bmk)

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


def run(params):

    args = candle.ArgumentStruct(**params)
    seed = args.rng_seed
    candle.set_seed(seed)

    # Construct extension to save model
    ext = p1b1.extension_from_parameters(params, '.keras')
    candle.verify_path(params['save_path'])
    prefix = '{}{}'.format(params['save_path'], ext)
    logfile = params['logfile'] if params['logfile'] else prefix + '.log'
    candle.set_up_logger(logfile, p1b1.logger, params['verbose'])
    p1b1.logger.info('Params: {}'.format(params))

    # Get default parameters for initialization and optimizer functions
    keras_defaults = candle.keras_default_config()

    # Load dataset
    x_train, y_train, x_val, y_val, x_test, y_test, x_labels, y_labels = p1b1.load_data(params, seed)

    # cache_file = 'data_l1000_cache.h5'
    # save_cache(cache_file, x_train, y_train, x_val, y_val, x_test, y_test, x_labels, y_labels)
    # x_train, y_train, x_val, y_val, x_test, y_test, x_labels, y_labels = load_cache(cache_file)

    p1b1.logger.info("Shape x_train: {}".format(x_train.shape))
    p1b1.logger.info("Shape x_val:   {}".format(x_val.shape))
    p1b1.logger.info("Shape x_test:  {}".format(x_test.shape))

    p1b1.logger.info("Range x_train: [{:.3g}, {:.3g}]".format(np.min(x_train), np.max(x_train)))
    p1b1.logger.info("Range x_val:   [{:.3g}, {:.3g}]".format(np.min(x_val), np.max(x_val)))
    p1b1.logger.info("Range x_test:  [{:.3g}, {:.3g}]".format(np.min(x_test), np.max(x_test)))

    p1b1.logger.debug('Class labels')
    for i, label in enumerate(y_labels):
        p1b1.logger.debug('  {}: {}'.format(i, label))

    # clf = build_type_classifier(x_train, y_train, x_val, y_val)

    # n_classes = len(y_labels)
    cond_train = y_train
    cond_val = y_val
    cond_test = y_test

    input_dim = x_train.shape[1]
    cond_dim = cond_train.shape[1]
    latent_dim = params['latent_dim']

    activation = params['activation']
    dropout = params['dropout']
    dense_layers = params['dense']
    dropout_layer = AlphaDropout if params['alpha_dropout'] else Dropout

    # Initialize weights and learning rule
    initializer_weights = candle.build_initializer(params['initialization'], keras_defaults, seed)
    initializer_bias = candle.build_initializer('constant', keras_defaults, 0.)

    if dense_layers is not None:
        if type(dense_layers) != list:
            dense_layers = list(dense_layers)
    else:
        dense_layers = []

    # Encoder Part
    x_input = Input(shape=(input_dim,))
    cond_input = Input(shape=(cond_dim,))
    h = x_input
    if params['model'] == 'cvae':
        h = keras.layers.concatenate([x_input, cond_input])

    for i, layer in enumerate(dense_layers):
        if layer > 0:
            x = h
            h = Dense(layer, activation=activation,
                      kernel_initializer=initializer_weights,
                      bias_initializer=initializer_bias)(h)
            if params['residual']:
                try:
                    h = keras.layers.add([h, x])
                except ValueError:
                    pass
            if params['batch_normalization']:
                h = BatchNormalization()(h)
            if dropout > 0:
                h = dropout_layer(dropout)(h)

    if params['model'] == 'ae':
        encoded = Dense(latent_dim, activation=activation,
                        kernel_initializer=initializer_weights,
                        bias_initializer=initializer_bias)(h)
    else:
        epsilon_std = params['epsilon_std']
        z_mean = Dense(latent_dim, name='z_mean')(h)
        z_log_var = Dense(latent_dim, name='z_log_var')(h)
        encoded = z_mean

        def vae_loss(x, x_decoded_mean):
            xent_loss = binary_crossentropy(x, x_decoded_mean)
            kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
            return K.mean(xent_loss + kl_loss / input_dim)

        def sampling(params):
            z_mean_, z_log_var_ = params
            batch_size = K.shape(z_mean_)[0]
            epsilon = K.random_normal(shape=(batch_size, latent_dim),
                                      mean=0., stddev=epsilon_std)
            return z_mean_ + K.exp(z_log_var_ / 2) * epsilon

        z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])
        # if params['model'] == 'cvae':
        #    z_cond = keras.layers.concatenate([z, cond_input])

    # Decoder Part
    decoder_input = Input(shape=(latent_dim,))
    h = decoder_input
    if params['model'] == 'cvae':
        h = keras.layers.concatenate([decoder_input, cond_input])

    for i, layer in reversed(list(enumerate(dense_layers))):
        if layer > 0:
            x = h
            h = Dense(layer, activation=activation,
                      kernel_initializer=initializer_weights,
                      bias_initializer=initializer_bias)(h)
            if params['residual']:
                try:
                    h = keras.layers.add([h, x])
                except ValueError:
                    pass
            if params['batch_normalization']:
                h = BatchNormalization()(h)
            if dropout > 0:
                h = dropout_layer(dropout)(h)

    decoded = Dense(input_dim, activation='sigmoid',
                    kernel_initializer=initializer_weights,
                    bias_initializer=initializer_bias)(h)

    # Build autoencoder model
    if params['model'] == 'cvae':
        encoder = Model([x_input, cond_input], encoded)
        decoder = Model([decoder_input, cond_input], decoded)
        model = Model([x_input, cond_input], decoder([z, cond_input]))
        loss = vae_loss
        metrics = [xent, corr, mse]
    elif params['model'] == 'vae':
        encoder = Model(x_input, encoded)
        decoder = Model(decoder_input, decoded)
        model = Model(x_input, decoder(z))
        loss = vae_loss
        metrics = [xent, corr, mse]
    else:
        encoder = Model(x_input, encoded)
        decoder = Model(decoder_input, decoded)
        model = Model(x_input, decoder(encoded))
        loss = params['loss']
        metrics = [xent, corr]

    model.summary()
    decoder.summary()

    if params['cp']:
        model_json = model.to_json()
        with open(prefix + '.model.json', 'w') as f:
            print(model_json, file=f)

    # Define optimizer
    # optimizer = candle.build_optimizer(params['optimizer'],
    #                                             params['learning_rate'],
    #                                             keras_defaults)
    optimizer = optimizers.deserialize({'class_name': params['optimizer'], 'config': {}})
    base_lr = params['base_lr'] or K.get_value(optimizer.lr)
    if params['learning_rate']:
        K.set_value(optimizer.lr, params['learning_rate'])

    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    # calculate trainable and non-trainable params
    params.update(candle.compute_trainable_params(model))

    def warmup_scheduler(epoch):
        lr = params['learning_rate'] or base_lr * params['batch_size'] / 100
        if epoch <= 5:
            K.set_value(model.optimizer.lr, (base_lr * (5 - epoch) + lr * epoch) / 5)
        p1b1.logger.debug('Epoch {}: lr={}'.format(epoch, K.get_value(model.optimizer.lr)))
        return K.get_value(model.optimizer.lr)

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001)
    warmup_lr = LearningRateScheduler(warmup_scheduler)
    checkpointer = ModelCheckpoint(params['save_path'] + ext + '.weights.h5', save_best_only=True, save_weights_only=True)
    tensorboard = TensorBoard(log_dir="tb/tb{}".format(ext))
    candle_monitor = candle.CandleRemoteMonitor(params=params)
    timeout_monitor = candle.TerminateOnTimeOut(params['timeout'])
    history_logger = LoggingCallback(p1b1.logger.debug)

    callbacks = [candle_monitor, timeout_monitor, history_logger]
    if params['reduce_lr']:
        callbacks.append(reduce_lr)
    if params['warmup_lr']:
        callbacks.append(warmup_lr)
    if params['cp']:
        callbacks.append(checkpointer)
    if params['tb']:
        callbacks.append(tensorboard)

    x_val2 = np.copy(x_val)
    np.random.shuffle(x_val2)
    start_scores = p1b1.evaluate_autoencoder(x_val, x_val2)
    p1b1.logger.info('\nBetween random pairs of validation samples: {}'.format(start_scores))

    if params['model'] == 'cvae':
        inputs = [x_train, cond_train]
        val_inputs = [x_val, cond_val]
        test_inputs = [x_test, cond_test]
    else:
        inputs = x_train
        val_inputs = x_val
        test_inputs = x_test

    outputs = x_train
    val_outputs = x_val
    # test_outputs = x_test

    history = model.fit(inputs, outputs,
                        verbose=2,
                        batch_size=params['batch_size'],
                        epochs=params['epochs'],
                        callbacks=callbacks,
                        validation_data=(val_inputs, val_outputs))

    if params['cp']:
        encoder.save(prefix + '.encoder.h5')
        decoder.save(prefix + '.decoder.h5')

    candle.plot_history(prefix, history, 'loss')
    candle.plot_history(prefix, history, 'corr', 'streaming pearson correlation')

    # Evalute model on test set
    x_pred = model.predict(test_inputs)
    scores = p1b1.evaluate_autoencoder(x_pred, x_test)
    p1b1.logger.info('\nEvaluation on test data: {}'.format(scores))

    x_test_encoded = encoder.predict(test_inputs, batch_size=params['batch_size'])
    y_test_classes = np.argmax(y_test, axis=1)
    candle.plot_scatter(x_test_encoded, y_test_classes, prefix + '.latent')

    if params['tsne']:
        tsne = TSNE(n_components=2, random_state=seed)
        x_test_encoded_tsne = tsne.fit_transform(x_test_encoded)
        candle.plot_scatter(x_test_encoded_tsne, y_test_classes, prefix + '.latent.tsne')

    # diff = x_pred - x_test
    # plt.hist(diff.ravel(), bins='auto')
    # plt.title("Histogram of Errors with 'auto' bins")
    # plt.savefig('histogram_keras.png')

    # generate synthetic data
    # epsilon_std = 1.0
    # for i in range(1000):
    #     z_sample = np.random.normal(size=(1, 2)) * epsilon_std
    #     x_decoded = decoder.predict(z_sample)

    p1b1.logger.handlers = []

    return history


def main():
    params = initialize_parameters()
    run(params)


if __name__ == '__main__':
    main()
    if K.backend() == 'tensorflow':
        K.clear_session()
