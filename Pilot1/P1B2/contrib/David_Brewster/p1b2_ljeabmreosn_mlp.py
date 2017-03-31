from __future__ import print_function

from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Input, advanced_activations
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint

from ..datasets import p1b2

import argparse

args = None
def set_args():
    global args
    parser = argparse.ArgumentParser(description='Variate some music.')
    parser.add_argument('--train', dest='training_file', help='input training data')
    parser.add_argument('--test', dest='testing_file', help='input testing data')
    parser.add_argument('--trainvar', dest='training_variable', help='variable you want to train')
    args = parser.parse_args()


BEST_MODEL_PATH = 'benchmarks/P1B2/best.hdf5'


(X_train, y_train), (X_test, y_test) = (None, None), (None, None)
input_dim = None
output_dim = None

def create_model():
    # advanced activation not used yet
    srelu = advanced_activations.SReLU(
        t_left_init='zero', 
        a_left_init='glorot_uniform', 
        t_right_init='glorot_uniform', 
        a_right_init='one'
    )

    # create and return model
    model = Sequential()
    model.add(Dense(256, input_dim=input_dim, activation='sigmoid'))
    model.add(Dense(256, activation='sigmoid'))
    model.add(Dense(output_dim, activation='sigmoid'))
    return model

def train_model(model):
    opt = 'rmsprop'
    model_checkpoint = ModelCheckpoint(
        filepath=BEST_MODEL_PATH, 
        monitor='val_acc', 
        verbose=0, 
        save_best_only=True,  
        mode='auto'
    )
    overfitting_stopper = EarlyStopping(
        monitor='val_acc', 
        min_delta=0, 
        patience=5, 
        verbose=1, 
        mode='auto'
    )
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model_history = model.fit(X_train, y_train,
	    batch_size       = 64,
	    nb_epoch         = 50,
	    #shuffle          = True,
	    validation_split = 0.2,
	    #verbose          = 2
        callbacks        = [overfitting_stopper, model_checkpoint]
    )

def load_model_from_path(weights_path):
    model = create_model()
    model.load_weights(weights_path)
    return model

def save_weights(model):
    for i, layer in enumerate(model.layers):
        weights = layer.get_weights()
        print('LAYER {}'.format(i))
        print(weights)
        print(weights.s)
        input()




##################### START OF PROGRAM ##############################
set_args()
# gather input
if args != None and args.training_file and args.testing_file and args.training_variable:
    try:
        (X_train, y_train), (X_test, y_test) = p1b2.load_data_from_file(
            train=args.training_file, 
            test=args.testing_file, 
            trainvar=args.training_variable)
    except Exception as e:
        print(e)
        print('(Hint: Are you sure the input files are in valid format?)')
else:
    (X_train, y_train), (X_test, y_test) = p1b2.load_data_from_url(n_cols=10000)

input_dim = X_train.shape[1]
output_dim = y_train.shape[1]

# train our model
train_model(create_model())
best_model = load_model_from_path(BEST_MODEL_PATH)
y_pred = best_model.predict(X_test)
accuracy = p1b2.evaluate(y_test, y_pred)

