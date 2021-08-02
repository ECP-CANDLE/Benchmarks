from __future__ import print_function

import numpy as np

from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.regularizers import l2

import p1b2
import candle


def initialize_parameters(default_model='p1b2_default_model.txt'):

    # Build benchmark object
    p1b2Bmk = p1b2.BenchmarkP1B2(p1b2.file_path, default_model, 'keras',
                                 prog='p1b2_baseline', desc='Train Classifier - Pilot 1 Benchmark 2')

    # Initialize parameters
    gParameters = candle.finalize_parameters(p1b2Bmk)

    return gParameters


def run(gParameters):

    # Construct extension to save model
    ext = p1b2.extension_from_parameters(gParameters, '.keras')
    candle.verify_path(gParameters['save_path'])
    prefix = '{}{}'.format(gParameters['save_path'], ext)
    logfile = gParameters['logfile'] if gParameters['logfile'] else prefix + '.log'
    candle.set_up_logger(logfile, p1b2.logger, gParameters['verbose'])
    p1b2.logger.info('Params: {}'.format(gParameters))

    # Get default parameters for initialization and optimizer functions
    kerasDefaults = candle.keras_default_config()
    seed = gParameters['rng_seed']

    # Load dataset
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = p1b2.load_data_one_hot(gParameters, seed)

    print("Shape X_train: ", X_train.shape)
    print("Shape X_val: ", X_val.shape)
    print("Shape X_test: ", X_test.shape)
    print("Shape y_train: ", y_train.shape)
    print("Shape y_val: ", y_val.shape)
    print("Shape y_test: ", y_test.shape)

    print("Range X_train --> Min: ", np.min(X_train), ", max: ", np.max(X_train))
    print("Range X_val --> Min: ", np.min(X_val), ", max: ", np.max(X_val))
    print("Range X_test --> Min: ", np.min(X_test), ", max: ", np.max(X_test))
    print("Range y_train --> Min: ", np.min(y_train), ", max: ", np.max(y_train))
    print("Range y_val --> Min: ", np.min(y_val), ", max: ", np.max(y_val))
    print("Range y_test --> Min: ", np.min(y_test), ", max: ", np.max(y_test))

    input_dim = X_train.shape[1]
    input_vector = Input(shape=(input_dim,))
    output_dim = y_train.shape[1]

    # Initialize weights and learning rule
    initializer_weights = candle.build_initializer(gParameters['initialization'], kerasDefaults, seed)
    initializer_bias = candle.build_initializer('constant', kerasDefaults, 0.)

    activation = gParameters['activation']

    # Define MLP architecture
    layers = gParameters['dense']

    if layers is not None:
        if type(layers) != list:
            layers = list(layers)
        for i, l in enumerate(layers):
            if i == 0:
                x = Dense(l, activation=activation,
                          kernel_initializer=initializer_weights,
                          bias_initializer=initializer_bias,
                          kernel_regularizer=l2(gParameters['reg_l2']),
                          activity_regularizer=l2(gParameters['reg_l2']))(input_vector)
            else:
                x = Dense(l, activation=activation,
                          kernel_initializer=initializer_weights,
                          bias_initializer=initializer_bias,
                          kernel_regularizer=l2(gParameters['reg_l2']),
                          activity_regularizer=l2(gParameters['reg_l2']))(x)
            if gParameters['dropout']:
                x = Dropout(gParameters['dropout'])(x)
        output = Dense(output_dim, activation=activation,
                       kernel_initializer=initializer_weights,
                       bias_initializer=initializer_bias)(x)
    else:
        output = Dense(output_dim, activation=activation,
                       kernel_initializer=initializer_weights,
                       bias_initializer=initializer_bias)(input_vector)

    # Build MLP model
    mlp = Model(outputs=output, inputs=input_vector)
    p1b2.logger.debug('Model: {}'.format(mlp.to_json()))

    # Define optimizer
    optimizer = candle.build_optimizer(gParameters['optimizer'],
                                       gParameters['learning_rate'],
                                       kerasDefaults)

    # Compile and display model
    mlp.compile(loss=gParameters['loss'], optimizer=optimizer, metrics=['accuracy'])
    mlp.summary()

    # Seed random generator for training
    np.random.seed(seed)

    mlp.fit(X_train, y_train,
            batch_size=gParameters['batch_size'],
            epochs=gParameters['epochs'],
            validation_data=(X_val, y_val)
            )

    # model save
    # save_filepath = "model_mlp_W_" + ext
    # mlp.save_weights(save_filepath)

    # Evalute model on test set
    y_pred = mlp.predict(X_test)
    scores = p1b2.evaluate_accuracy_one_hot(y_pred, y_test)
    print('Evaluation on test data:', scores)


def main():
    params = initialize_parameters()
    run(params)


if __name__ == '__main__':
    main()
    try:
        K.clear_session()
    except AttributeError:      # theano does not have this function
        pass
