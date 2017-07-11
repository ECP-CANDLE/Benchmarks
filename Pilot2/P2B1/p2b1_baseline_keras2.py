from __future__ import print_function

import numpy as np

#import argparse

from keras import backend as K

from keras.layers import Input, Dense, Dropout, Activation, Conv1D, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, Convolution1D
from keras.optimizers import SGD, Adam, RMSprop
from keras.models import Sequential, Model#, model_from_json, model_from_yaml
from keras.regularizers import l2
from keras.callbacks import LearningRateScheduler
from keras import callbacks

#from keras.utils import np_utils
#from keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau

import p2b1 as benchmark
import default_utils
import keras_utils

from solr_keras import CandleRemoteMonitor, compute_trainable_params, TerminateOnTimeOut


def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")


##### Define Neural Network Models ###################
def dense_auto(weights_path=None, input_shape=(784,),
                hidden_layers=None, nonlinearity='relu', l2_reg=0.0):
    
    input_img = Input(shape=input_shape)

    if hidden_layers!=None:
        if type(hidden_layers)!=list:
            hidden_layers=list(hidden_layers)
        for i,l in enumerate(hidden_layers):
            if i==0:
                encoded=Dense(l,activation=nonlinearity,kernel_regularizer=l2(l2_reg))(input_img)
            else:
                encoded=Dense(l,activation=nonlinearity,kernel_regularizer=l2(l2_reg))(encoded)

        for i,l in reversed(list(enumerate(hidden_layers))):
            if i <len(hidden_layers)-1:
                if i==len(hidden_layers)-2:
                    decoded=Dense(l,activation=nonlinearity,kernel_regularizer=l2(l2_reg))(encoded)
                else:
                    decoded=Dense(l,activation=nonlinearity,kernel_regularizer=l2(l2_reg))(decoded)
        decoded=Dense(input_shape[0],kernel_regularizer=l2(l2_reg))(decoded)
    else:
        decoded=Dense(input_shape[0],kernel_regularizer=l2(l2_reg))(input_img)

    model=Model(outputs=decoded,inputs=input_img)

    if weights_path:
        print('Loading Model')
        model.load_weights(weights_path)
    return model

def conv_dense_auto(weights_path=None, input_shape=(1,784),
                    hidden_layers=None, nonlinearity='relu', l2_reg=0.0):
    
    kernel_size=7
    input_img = Input(shape=input_shape)

    if hidden_layers!=None:
        if type(hidden_layers)!=list:
            hidden_layers=list(hidden_layers)
        for i,l in enumerate(hidden_layers):
            if i==0:
                encoded=Convolution1D(l,kernel_size,padding='same',input_shape=input_shape,activation=nonlinearity,kernel_regularizer=l2(l2_reg))(input_img)
            else:
                encoded=Convolution1D(l,kernel_size,padding='same',input_shape=input_shape,activation=nonlinearity,kernel_regularizer=l2(l2_reg))(encoded)

        encoded=Flatten()(encoded) ## reshape output of 1d convolution layer

        for i,l in reversed(list(enumerate(hidden_layers))):
            if i <len(hidden_layers)-1:
                if i==len(hidden_layers)-2:
                    decoded=Dense(l,activation=nonlinearity,kernel_regularizer=l2(l2_reg))(encoded)
                else:
                    decoded=Dense(l,activation=nonlinearity,kernel_regularizer=l2(l2_reg))(decoded)
        decoded=Dense(input_shape[1],kernel_regularizer=l2(l2_reg))(decoded)

    else:
        decoded=Dense(input_shape[1],kernel_regularizer=l2(l2_reg))(input_img)

    model=Model(inputs=input_img,outputs=decoded)

    if weights_path:
        print('Loading Model')
        model.load_weights(weights_path)
    return model

##### ############################################## ###################

def initialize_parameters():

    # Build benchmark object
    p2b1Bmk = benchmark.BenchmarkP2B1(benchmark.file_path, 'p2b1_default_model.txt', 'keras',
    prog='p2b1_baseline', desc='Train Molecular Frame Autoencoder - Pilot 2 Benchmark 1')
    
    # Initialize parameters
    gParameters = default_utils.initialize_parameters(p2b1Bmk)
    #benchmark.logger.info('Params: {}'.format(gParameters))

    return gParameters


def load_data(gParameters):

    (data_files, fields) = benchmark.get_list_of_data_files(gParameters)

    ## Define datagenerator
    datagen = benchmark.ImageNoiseDataGenerator(corruption_level=gParameters['noise_factor'])

    ## get data dimension ##
    num_samples = 0
    for f in data_files[0:1]:
        X = np.load(f)
        num_samples += X.shape[0]

    X = np.load(data_files[0])
    print ("X shape: ", X.shape)

    molecular_hidden_layers = gParameters['molecular_num_hidden']

    X_train = []
    if not molecular_hidden_layers:
        X_train = benchmark.get_data(X, case=gParameters['case'])
        input_dim = X_train.shape[1]
    else:
        ## computing input dimension for outer AE
        input_dim = X.shape[1]*molecular_hidden_layers[-1]

    print ('The input dimension is ', input_dim)
    

    ## get data dimension for molecular autoencoder
    if not gParameters['type_bool']:
        molecular_input_dim=np.prod([X.shape[2],X.shape[3]-5])## only consider molecular location coordinates
        molecular_output_dim=np.prod([X.shape[2],X.shape[3]-5])
    else:
        molecular_input_dim=np.prod(X.shape[2:])
        molecular_output_dim=np.prod(X.shape[2:])

    print ('Data Format:\n  [Frames (%s), Molecules (%s), Beads (%s), %s (%s)]' % (
        num_samples, X.shape[1], X.shape[2], fields.keys(), X.shape[3]))

    
    return (X, num_samples, X_train, input_dim, molecular_input_dim), datagen, data_files


def build_model(gParameters, kerasDefaults, data):

    # unfold data
    X, num_samples, X_train, input_dim, molecular_input_dim = data


    ### Define Model, Solver and Compile ##########
    print ('Define the model and compile')
    # Define optimizer
    optimizer = keras_utils.build_optimizer(gParameters['optimizer'],
                                            gParameters['learning_rate'],
                                            kerasDefaults)

    print ('using mlp network')
    model_type='mlp'
    hidden_layers = gParameters['dense']
    model = dense_auto(weights_path=gParameters['weight_path'],
                        input_shape=(input_dim,),nonlinearity=gParameters['activation'],
                        hidden_layers=hidden_layers,l2_reg=gParameters['weight_decay'])

    print ('Autoencoder Regression problem')
    model.compile(optimizer=optimizer, loss=gParameters['loss'])
    
    #### Print Model Stats ###########
    #KEU.Model_Info(model)
    model.summary()

    ######## Define Molecular Model, Solver and Compile #########
    molecular_nonlinearity = gParameters['molecular_activation']
    molecular_hidden_layers = gParameters['molecular_num_hidden']

    len_molecular_hidden_layers = len(molecular_hidden_layers)
    conv_bool = gParameters['conv_bool']
    if conv_bool:
        molecular_model = conv_dense_auto(weights_path=None,
                            input_shape=(1,molecular_input_dim),
                            nonlinearity=molecular_nonlinearity,
                            hidden_layers=molecular_hidden_layers,
                            l2_reg=gParameters['weight_decay'])
    else:
        molecular_model = dense_auto(weights_path=None,
                                    input_shape=(molecular_input_dim,),
                                    nonlinearity=molecular_nonlinearity,
                                    hidden_layers=molecular_hidden_layers,
                                    l2_reg=gParameters['weight_decay'])

    molecular_model.compile(optimizer=optimizer, loss=gParameters['loss'], metrics=[gParameters['metrics']])
    molecular_model.summary()

    return model, molecular_model


def train_model(gParameters,
                datagen, data_files,
                model, molecular_model):


    ##### set up callbacks and cooling for the molecular_model ##########
    drop=0.5
    mb_epochs = gParameters['molecular_epochs']
    initial_lrate = gParameters['learning_rate']
    epochs_drop = 1 + int(np.floor(mb_epochs/3))
    
    def step_decay(epoch):
        global initial_lrate,epochs_drop,drop
        lrate = initial_lrate * np.power(drop, np.floor((1+epoch)/epochs_drop))
        return lrate
    
    lr_scheduler = LearningRateScheduler(step_decay)
    history = callbacks.History()
    #callbacks=[history,lr_scheduler]
    gParameters.update(compute_trainable_params(model))
    
    candleRemoteMonitor = CandleRemoteMonitor(params=gParameters)
    timeoutMonitor = TerminateOnTimeOut(gParameters['timeout'])
    callbacks_=[history, candleRemoteMonitor, timeoutMonitor]
    loss = 0.
    
    len_molecular_hidden_layers = len(gParameters['molecular_num_hidden'])

    
    #### Train the Model
    if gParameters['train_bool']:
        if not gParameters['cool']:
            effec_epochs = gParameters['epochs']
            ct = benchmark.Candle_Composite_Train(datagen, model, molecular_model,
                                data_files, mb_epochs, effec_epochs,
                                callbacks_, batch_size=32, case=gParameters['case'],
                                scale_factor=0.5, len_molecular_hidden_layers=len_molecular_hidden_layers,
                                conv_bool=gParameters['conv_bool'], type_bool=gParameters['type_bool'])
#            ct=hf.Candle_Train(datagen,model,data_files,effec_epochs,case=GP['case'])
            loss = ct.train_ac()
        else:
            effec_epochs = gParameters['epochs']//3
            ct = benchmark.Candle_Train(datagen, model, data_files, effec_epochs, case=gParameters['case'])
            loss = []
            for i in range(3):
                lr = gParameters['learning_rate']/10**i
                ct.model.optimizer.lr.set_value(lr)
                if i > 0:
                    ct.print_data=False
                    print ('Cooling Learning Rate by factor of 10...')
                loss.extend(ct.train_ac())


        if False and gParameters['output_dir']!=None:
            # Not necessary -> path already built
            #if not os.path.exists(gParameters['output_dir']):
            #    os.makedirs(gParameters['output_dir'])

            loss_file='%s/%s_%s.pkl'%(gParameters['output_dir'], gParameters['model_name'], 'mlp')
            o=open(loss_file,'wb')
            pickle.dump(loss,o)
            o.close()
            
            model.save_weights("{}/{}.model.h5".format(gParameters['output_dir'], gParameters['model_name']))
            print("Saved model to disk")

    return loss



def run(gParameters, data):

    # Get default parameters for initialization and optimizer functions
    kerasDefaults = default_utils.keras_default_config()

    dat, datagen, data_files = data

    # build model
    model, molecular_model = build_model(gParameters, kerasDefaults, dat)

    # train model
    loss = train_model(gParameters, datagen, data_files,
                model, molecular_model)

    return loss



def main():

    gParameters = initialize_parameters()
    data = load_data(gParameters)
    run(gParameters, data)



if __name__ == '__main__':
    main()
    try:
        K.clear_session()
    except AttributeError:      # theano does not have this function
        pass
