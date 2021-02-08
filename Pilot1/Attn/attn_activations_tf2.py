from tensorflow.keras import utils, Input, Model
from tensorflow.keras.metrics import AUC
from tensorflow.keras.layers import concatenate, multiply
from tensorflow.keras.layers import Dense, Dropout, Activation, BatchNormalization
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import CSVLogger, ReduceLROnPlateau
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.utils.class_weight import compute_class_weight
from keract import get_activations
import numpy as np
import pandas as pd
import os, sys

file_path = os.path.dirname(os.path.realpath(__file__))
lib_path2 = os.path.abspath(os.path.join(file_path, '..', '..', 'common'))
sys.path.append(lib_path2)
import candle
import attn

sys.path.append("../../../svcca")
import cca_core


# build the model (this can come from attn_baseline_keras2.py)
def build_model(params, PS=0):
    inputs = Input(shape=(PS,))
    x = Dense(params['dense'][0], activation=params['activation'][0])(inputs)
    x = BatchNormalization()(x)
    a = Dense(params['dense'][1], activation=params['activation'][1])(x)
    a = BatchNormalization()(a)
    b = Dense(params['dense'][2], activation=params['activation'][2])(x)
    x = multiply([a, b])

    for i in range(3, len(params['dense']) - 1):
        x = Dense(params['dense'][i], activation=params['activation'][i])(x)
        x = BatchNormalization()(x)
        x = Dropout(params['dropout'])(x)

    outputs = Dense(params['dense'][-1], activation=params['activation'][-1])(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.summary()
    return model

def initialize_parameters(default_model='attn_default_model.txt'):
    attnBmk = attn.BenchmarkAttn(
            attn.file_path,
            default_model,
            'keras',
            prog='attn_activations',
            desc='Fully connected networks with attention for experimantal use'
            )
    gParameters = candle.finalize_parameters(attnBmk)
    if 'sample_size' in gParameters:
        gParameters['sample_size'] = int(gParameters['sample_size'])
    if 'training_size' in gParameters:
        gParameters['training_size'] = int(gParameters['training_size'])

    return gParameters

def subset_data (X_train, Y_train, params):
    if params['training_size'] > 0:
        if params['training_size'] > X_train.shape[0]:
            print ('setting training_size to {}'. format(X_train.shape[0]))
            return X_train, Y_train
        else:
            X_train['_Y1'] = Y_train[:,0]
            X_train['_Y2'] = Y_train[:,1]

            _X_train = pd.DataFrame()
            _X_train = X_train.sample(n=params['training_size'], replace=False,
                    random_state=params['rng_seed'], axis=0)
            _Y_train = _X_train[['_Y1','_Y2']].to_numpy()

            X_train.drop(['_Y1','_Y2'], axis=1, inplace=True)
            _X_train.drop(['_Y1','_Y2'], axis=1, inplace=True)

    return _X_train, _Y_train

# start run method
# start run
def run(params):
    args = candle.ArgumentStruct(**params)
    seed = args.rng_seed
    candle.set_seed(seed)

        # Construct extension to save model
    ext = attn.extension_from_parameters(params, 'keras')
    candle.verify_path(params['save_path'])
    prefix = '{}{}'.format(params['save_path'], ext)
    logfile = params['logfile'] if params['logfile'] else prefix + '.log'
    root_fname = 'Agg_attn_bin'
    candle.set_up_logger(logfile, attn.logger, params['verbose'])
    attn.logger.info('Params: {}'.format(params))

    # Get default parameters for initialization and optimizer functions
    keras_defaults = candle.keras_default_config()

    # load data (this can come from attn.py)
    # train_file = "../../Benchmarks/Data/Pilot1/top_21_1fold_001.h5"
    # print('processing h5 in file {}'.format(train_file))
    X_train, _Y_train, X_val, _Y_val, X_test, _Y_test = attn.load_data(
            params,
            params['rng_seed']
            )

    Y_train = _Y_train['AUC']
    Y_test = _Y_test['AUC']
    Y_val = _Y_val['AUC']

    # convert classes to integers
    nb_classes = params['dense'][-1]
    Y_train = utils.to_categorical(Y_train, nb_classes)
    Y_test = utils.to_categorical(Y_test, nb_classes)
    Y_val = utils.to_categorical(Y_val, nb_classes)

    # compute class weights
    y_integers = np.argmax(Y_train, axis=1)
    class_weights = compute_class_weight('balanced', np.unique(y_integers), y_integers)
    d_class_weights = dict(enumerate(class_weights))

    model_1 = build_model(params, PS=X_train.shape[1])
    model_2 = build_model(params, PS=X_train.shape[1])

    # compile the models
    model_1.compile(loss=params['loss'],
                    optimizer=params['optimizer'],
                    metrics=['acc', AUC(name='tf_auc')])

    model_2.compile(loss=params['loss'],
                    optimizer=params['optimizer'],
                    metrics=['acc', AUC(name='tf_auc')])

    # train the models
    callbacks = []

    reduce_lr = ReduceLROnPlateau(
            monitor='val_tf_auc',
            factor=0.20,
            patience=40,
            verbose=1,
            mode='auto',
            min_delta=0.0001,
            cooldown=3,
            min_lr=0.000000001
            )
    early_stop = EarlyStopping(
            monitor='val_tf_auc',
            patience=200,
            verbose=1,
            mode='auto'
            )
    callbacks = [reduce_lr, early_stop]

    #os.environ["CUDA_VISIBLE_DEVICES"]="0";
    X_train, Y_train = subset_data(X_train, Y_train, params)
    print('sampled training data size {} and labels {}'.format(_X_train.shape,_Y_train.shape))
    history_1 = model_1.fit(X_train, Y_train, class_weight=d_class_weights,
                                batch_size=params['batch_size'],
                                epochs=params['epochs'],
                                verbose=1,
                                validation_data=(X_val, Y_val),
                                callbacks=callbacks)

    history_2 = model_2.fit(X_train, Y_train, class_weight=d_class_weights,
                                batch_size=params['batch_size'],
                                epochs=params['epochs'],
                                verbose=1,
                                validation_data=(X_val, Y_val),
                                callbacks=callbacks)


    # get activations on those data points, 50000 is 5x (num_neurons in the widest hidden layer)
    # x = X_test.iloc[1:ARG1].to_numpy()
    _i = int(params['sample_size'])  ## JAMAL, why is params['sample_size'] not an int?
    x = X_test.iloc[1:_i].to_numpy()
    activations_1 = get_activations(model_1, x, auto_compile=True)
    activations_2 = get_activations(model_2, x, auto_compile=True)

    # make the first dim the neurons, the second dim the activations, one activation per sample per neuron
    act1 = pd.DataFrame(activations_1['dense_1']).transpose()
    print (act1.shape)

    act2= pd.DataFrame(activations_2['dense_10']).transpose()
    print (act2.shape)

    act1 = act1.replace([np.inf, -np.inf], np.nan)
    act1 = act1.dropna(how="any").to_numpy()
    print (act1.shape)

    act2 = act2.replace([np.inf, -np.inf], np.nan)
    act2 = act2.dropna(how="any").to_numpy()
    print (act2.shape)

    results = cca_core.get_cca_similarity(act1, act2, verbose=True)
    print('Single number for summarizing similarity of pair {} is {:.4f}'.format(
        os.getpid(),
        np.mean(results["cca_coef1"]))
        )

    return history_1, history_2

def main():
    params = initialize_parameters()
    run(params)


if __name__ == '__main__':
    main()
    #if K.backend() == 'tensorflow':
    #    K.clear_session()
