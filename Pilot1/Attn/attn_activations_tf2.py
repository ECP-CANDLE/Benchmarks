from tensorflow.keras import utils, Input, Model
from tensorflow.keras.metrics import AUC
from tensorflow.keras.layers import concatenate, multiply
from tensorflow.keras.layers import Dense, Dropout, Activation, BatchNormalization
from tensorflow.keras.initializers import GlorotNormal
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import CSVLogger, ReduceLROnPlateau
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import model_from_json, model_from_yaml

from sklearn.utils.class_weight import compute_class_weight
from keract import get_activations
import numpy as np
import pandas as pd
import os, sys
from pathlib import Path

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
    x = Dense(params['dense'][0],
            kernel_initializer=GlorotNormal(seed=np.random.randint(1,1000)),
            activation=params['activation'][0])(inputs)
    x = BatchNormalization()(x)
    a = Dense(params['dense'][1],
            kernel_initializer=GlorotNormal(seed=np.random.randint(1,1000)),
            activation=params['activation'][1])(x)
    a = BatchNormalization()(a)
    b = Dense(params['dense'][2],
            kernel_initializer=GlorotNormal(seed=np.random.randint(1,1000)),
            activation=params['activation'][2])(x)
    x = multiply([a, b])

    for i in range(3, len(params['dense']) - 1):
        x = Dense(params['dense'][i],
                kernel_initializer=GlorotNormal(seed=np.random.randint(1,1000)),
                activation=params['activation'][i])(x)
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
    if 'training_size' in params and params['training_size'] > 0:
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
    else:
        _X_train, _Y_train = X_train, Y_train

    return _X_train, _Y_train

def save_and_test_saved_model(params, model, root_fname, X_train, X_test, Y_test):
    save_path = os.path.join(params['save_path'], '')
    if not os.path.exists(save_path):
        #os.mkdir(save_path)
        Path.mkdir(save_path, parents=True, exist_ok=True)

    # serialize model to JSON
    model_json = model.to_json()
    with open(save_path + root_fname + ".model.json", "w") as json_file:
        json_file.write(model_json)

    # serialize model to YAML
    model_yaml = model.to_yaml()
    with open(save_path + root_fname + ".model.yaml", "w") as yaml_file:
        yaml_file.write(model_yaml)

    # serialize weights to HDF5
    model.save_weights(save_path + root_fname + ".model.h5")
    print("Saved model to disk")

    # load json and create model
    json_file = open(save_path + root_fname + '.model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model_json = model_from_json(loaded_model_json)

    # load yaml and create model
    yaml_file = open(save_path + root_fname + '.model.yaml', 'r')
    loaded_model_yaml = yaml_file.read()
    yaml_file.close()
    loaded_model_yaml = model_from_yaml(loaded_model_yaml)

    # load weights into new model
    loaded_model_json.load_weights(
            save_path + root_fname + ".model.h5"
            )
    print("Loaded json model from disk")

    # evaluate json loaded model on test data
    loaded_model_json.compile(
            loss='binary_crossentropy',
            optimizer=params['optimizer'],
            metrics=['accuracy']
            )
    score_json = loaded_model_json.evaluate(X_test, Y_test, verbose=0)

    print('json Validation loss:', score_json[0])
    print('json Validation accuracy:', score_json[1])

    print("json %s: %.2f%%" % (
        loaded_model_json.metrics_names[1], score_json[1] * 100)
        )

    # load weights into new model
    loaded_model_yaml.load_weights(
            save_path + root_fname + ".model.h5"
            )
    print("Loaded yaml model from disk")

    # evaluate loaded model on test data
    loaded_model_yaml.compile(
            loss='binary_crossentropy',
            optimizer=params['optimizer'],
            metrics=['accuracy']
            )
    score_yaml = loaded_model_yaml.evaluate(X_test, Y_test, verbose=0)

    print('yaml Validation loss:', score_yaml[0])
    print('yaml Validation accuracy:', score_yaml[1])
    print("yaml %s: %.2f%%" % (
        loaded_model_yaml.metrics_names[1], score_yaml[1] * 100)
        )

    # predict using loaded yaml model on test and training data
    predict_yaml_train = loaded_model_yaml.predict(X_train)
    predict_yaml_test = loaded_model_yaml.predict(X_test)

    print('Yaml_train_shape:', predict_yaml_train.shape)
    print('Yaml_test_shape:', predict_yaml_test.shape)

    predict_yaml_train_classes = np.argmax(predict_yaml_train, axis=1)
    predict_yaml_test_classes = np.argmax(predict_yaml_test, axis=1)
    np.savetxt(
            save_path + root_fname + "_predict_yaml_train.csv",
            predict_yaml_train,
            delimiter=",",
            fmt="%.3f"
            )
    np.savetxt(
            save_path + root_fname + "_predict_yaml_test.csv",
            predict_yaml_test,
            delimiter=",",
            fmt="%.3f")

    np.savetxt(
            save_path + root_fname + "_predict_yaml_train_classes.csv",
            predict_yaml_train_classes,
            delimiter=",",
            fmt="%d"
            )
    np.savetxt(
            save_path + root_fname + "_predict_yaml_test_classes.csv",
            predict_yaml_test_classes,
            delimiter=",",
            fmt="%d"
            )

# start run
def run(params):
    args = candle.ArgumentStruct(**params)
    seed = args.rng_seed
    candle.set_seed(seed)

    if 'cca_epsilon' in params:
        cca_epsilon = params['cca_epsilon']
    else:
        cca_epsilon = 1e-10
     
    # Construct extension to save model
    ext = attn.extension_from_parameters(params, 'keras')
    candle.verify_path(params['save_path'])
    prefix = '{}{}'.format(params['save_path'], ext)
    logfile = params['logfile'] if params['logfile'] else prefix + '.log'
    candle.set_up_logger(logfile, attn.logger, params['verbose'])
    attn.logger.info('Params: {}'.format(params))

    # Get default parameters for initialization and optimizer functions
    keras_defaults = candle.keras_default_config()

    # Load the data
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

    # Build the models
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
    print('sampled training data size {} and labels {}'.format(X_train.shape,Y_train.shape))
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
    if 'sample_size' in params:
        x = X_test.iloc[1:params['sample_size']].to_numpy()
    else:
        x = X_test.to_numpy()
    activations_1 = get_activations(model_1, x, auto_compile=True)
    activations_2 = get_activations(model_2, x, auto_compile=True)

    for i in range(0, len(model_1.layers)):
        print ('layer {} with name {}'.format(i, model_1.layers[i].name))
        if 'dense' in model_1.layers[i].name:
            results = compute_cca(i, model_1, model_2, x, epsilon=cca_epsilon)
            svcca_results = compute_svcca(i, model_1, model_2, x, epsilon=cca_epsilon)

            print('Mean CCA Coef and SVCCA Coef of pair {} with training size {} at layer {} after {} epochs is {:.4f} and {:4f}'.format(
                os.getpid(),
                params['training_size'],
                i,
                params['epochs'],
                np.mean(results["cca_coef1"]),
                np.mean(svcca_results["cca_coef1"]))
                )


    save_and_test_saved_model(params, model_1, 'attn_1', X_train, X_test, Y_test)
    save_and_test_saved_model(params, model_2, 'attn_2', X_train, X_test, Y_test)

    return history_1, history_2

def compute_cca(i, model_1, model_2, x, epsilon=0):

    # get activations. This is inefficient when compute_cca is called in a loop
    # another variant would be to pass in layer names and activations

    activations_1 = get_activations(model_1, x, auto_compile=True)
    activations_2 = get_activations(model_2, x, auto_compile=True)
    print('{}\t{}'.format(i, model_1.layers[i].name))

    act1 = pd.DataFrame(activations_1[model_1.layers[i].name]).transpose()
    act2= pd.DataFrame(activations_2[model_2.layers[i].name]).transpose()
    print (act1.shape)
    print (act2.shape)

    act1 = act1.replace([np.inf, -np.inf], np.nan)
    act1 = act1.dropna(how="any").to_numpy()
    act2 = act2.replace([np.inf, -np.inf], np.nan)
    act2 = act2.dropna(how="any").to_numpy()
    print (act1.shape)
    print (act2.shape)

    results = cca_core.get_cca_similarity(act1, act2, verbose=True, epsilon=epsilon)
    return results

def compute_svcca (i, model_1, model_2, x, epsilon=0):

    activations_1 = get_activations(model_1, x, auto_compile=True)
    activations_2 = get_activations(model_2, x, auto_compile=True)
    print('{}\t{}'.format(i, model_1.layers[i].name))

    act1 = pd.DataFrame(activations_1[model_1.layers[i].name]).transpose()
    act2= pd.DataFrame(activations_2[model_2.layers[i].name]).transpose()
    print (act1.shape)
    print (act2.shape)

    act1 = act1.replace([np.inf, -np.inf], np.nan)
    act1 = act1.dropna(how="any").to_numpy()
    act2 = act2.replace([np.inf, -np.inf], np.nan)
    act2 = act2.dropna(how="any").to_numpy()
    print (act1.shape)
    print (act2.shape)

    # Mean subtract activations
    cact1 = act1 - np.mean(act1, axis=1, keepdims=True)
    cact2 = act2 - np.mean(act2, axis=1, keepdims=True)

    # Perform SVD
    U1, s1, V1 = np.linalg.svd(cact1, full_matrices=False)
    U2, s2, V2 = np.linalg.svd(cact2, full_matrices=False)

    num_sv = 20
    if cact1.shape[0] >= num_sv:
        svacts1 = np.dot(s1[:num_sv]*np.eye(num_sv), V1[:num_sv])
        svacts2 = np.dot(s2[:num_sv]*np.eye(num_sv), V2[:num_sv])
        svcca_results = cca_core.get_cca_similarity(svacts1, svacts2, epsilon=1e-10, verbose=False)

    else:
        print('number of rows {} in activations is less than number of singular vectors {}'.format(
            cact1.shape[0], num_sv))

        num_sv = cact1.shape[0]
        svacts1 = np.dot(s1[:num_sv]*np.eye(num_sv), V1[:num_sv])
        svacts2 = np.dot(s2[:num_sv]*np.eye(num_sv), V2[:num_sv])
        svcca_results = cca_core.get_cca_similarity(svacts1, svacts2, epsilon=1e-10, verbose=False)

    return svcca_results


def main():
    params = initialize_parameters()
    run(params)


if __name__ == '__main__':
    main()
    #if K.backend() == 'tensorflow':
    #    K.clear_session()
