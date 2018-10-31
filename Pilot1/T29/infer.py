import pandas as pd
import numpy as np
import os
import sys
import keras as ke
from keras.models import Sequential, Model, model_from_json, model_from_yaml
from keras.utils import np_utils
from keras import backend as K
from keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, LearningRateScheduler
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler


file_path = os.path.dirname(os.path.realpath(__file__))

# candle
sys.path.append('/raid/brettin/Benchmarks/common')
import candle_keras

# This needs to be fixed
# candle
def initialize_parameters():
    t29_common = candle_keras.Benchmark(file_path, 't29_default_model.txt','keras',
                            prog='t29res.py',desc='resnet')

    # Need a pointer to the docs showing what is provided
    # by default
    additional_definitions = [
        {'name':'connections',
         'default':1,
         'type':int,
         'help':'The number of residual connections.'},
        {'name':'distance',
         'default':1,
         'type':int,
         'help':'Residual connection distance between dense layers.'},
        {'name':'model',
         'default':'model.json',
         'type':str,
         'help':'Name of json model description file.'},
        {'name':'weights',
         'default':'model.h5',
         'type':str,
         'help':'Name of h5 weights file.'},
        {'name':'n_pred',
         'default':1,
         'type':int,
         'help':'Number of predictions to do on each sample.'}
    ]
    t29_common.additional_definitions = additional_definitions
    gParameters = candle_keras.initialize_parameters(t29_common)
    return gParameters


def load_data(gParameters):
    train_path=gParameters['train_path']
    test_path=gParameters['test_path']
    df_train = (pd.read_csv(train_path,header=None).values).astype('float32')
    df_test = (pd.read_csv(test_path,header=None).values).astype('float32')

    print('df_train shape:', df_train.shape)
    print('df_test shape:', df_test.shape)

    df_y_train = df_train[:,0].astype('int')
    df_y_test = df_test[:,0].astype('int')

    Y_train = np_utils.to_categorical(df_y_train,gParameters['classes'])
    train_classes = np.argmax(Y_train, axis=1)
    
    Y_test = np_utils.to_categorical(df_y_test,gParameters['classes'])
    test_classes = np.argmax(Y_test, axis=1)
              
    df_x_train = df_train[:, 1:df_train.shape[1]].astype(np.float32)
    df_x_test = df_test[:, 1:df_train.shape[1]].astype(np.float32)
            
    # not sure the extra variable is needed, and is this a copy or reference
    X_train = df_x_train
    X_test = df_x_test
            
    scaler = MaxAbsScaler()
    mat = np.concatenate((X_train, X_test), axis=0)
    mat = scaler.fit_transform(mat)
       
    X_train = mat[:X_train.shape[0], :]
    X_test = mat[X_train.shape[0]:, :]
        
    return X_train, Y_train, X_test, Y_test

# This is required for candle compliance.
# It essentially wraps what was in the implicit main funcion
def run(gParameters):
    print ('gParameters: ', gParameters)

    # load the data
    X_train, Y_train, X_test, Y_test = load_data(gParameters)

    # load json and create model
    json_file = open(gParameters['model'], 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model_json = model_from_json(loaded_model_json)

    # load weights into new model
    loaded_model_json.load_weights(gParameters['weights'])
    print("Loaded json model from disk")

    # predict using loaded yaml model on test and training data
    pred_test_df = pd.DataFrame()
    pred_test_classes_df = pd.DataFrame()

    for x in range(gParameters['n_pred']):
        predict_test = loaded_model_json.predict(X_test)
        pred_test_df[str(x)] = np.amax(predict_test, axis=1)
        pred_test_classes_df[str(x)] = np.argmax(predict_test, axis=1)

    pred_test_df['mean'] = pred_test_df.mean(axis=1)
    pred_test_df['std'] = pred_test_df.std(axis=1)

    pred_test_df.to_csv("predict_test.csv")
    pred_test_classes_df.to_csv("predict_test_classes.csv")
    return

# This is also added for candle compliance so that the program can
# still be executed independently from the command line.
def main():

    gParameters = initialize_parameters()
    run(gParameters)

if __name__ == '__main__':
    main()
    try:
        ke.clear_session()
    except AttributeError:      # theano does not have this function
        pass

