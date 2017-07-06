import numpy as np
import scipy as sp
import pickle
import sys,os
import argparse

TIMEOUT=3600 # in sec; set this to -1 for no timeout 
file_path = os.path.dirname(os.path.realpath(__file__))
lib_path = os.path.abspath(os.path.join(file_path, '..', 'common'))
sys.path.append(lib_path)
lib_path2 = os.path.abspath(os.path.join(file_path, '..','..', 'common'))
sys.path.append(lib_path2)

from keras import backend as K

from data_utils import get_file

import p2b1 as p2b1
import p2_common as p2c
import p2_common_keras as p2ck
from solr_keras import CandleRemoteMonitor, compute_trainable_params, TerminateOnTimeOut

HOME=os.environ['HOME']
def parse_list(option, opt, value, parser):
  setattr(parser.values, option.dest, value.split(','))
def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")

def get_p2b1_parser():
        parser = argparse.ArgumentParser(prog='p2b1_baseline',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            description='Train Molecular Frame Autoencoder - Pilot 2 Benchmark 1')

        return p2b1.common_parser(parser)

def initialize_parameters():

    parser = get_p2b1_parser()
    args = parser.parse_args()
    print('Args', args)

    GP=p2b1.read_config_file(args.config_file)
    print (GP)

    GP = p2c.args_overwrite_config(args, GP)

    if GP['backend'] != 'theano' and GP['backend'] != 'tensorflow':
      sys.exit('Invalid backend selected: %s' % GP['backend'])

    os.environ['KERAS_BACKEND'] = GP['backend']
    reload(K)
    if GP['backend'] == 'theano':
      K.set_image_dim_ordering('th')
    elif GP['backend'] == 'tensorflow':
      K.set_image_dim_ordering('tf')

    return GP

def run(GP):

    ## set the seed
    if GP['seed']:
    	np.random.seed(7)
    else:
    	np.random.seed(np.random.randint(10000))

    ## Set paths
    if not os.path.isdir(GP['home_dir']):
    	print ('Keras home directory not set')
    	sys.exit(0)
    sys.path.append(GP['home_dir'])

    import p2b1 as hf
    reload(hf)

    import keras_model_utils as KEU
    reload(KEU)
    reload(p2ck)
    reload(p2ck.optimizers)
    maps=hf.autoencoder_preprocess()

    from keras.optimizers import SGD,RMSprop,Adam
    from keras.datasets import mnist
    from keras.callbacks import LearningRateScheduler,ModelCheckpoint
    from keras import callbacks
    from keras.layers.advanced_activations import ELU
    from keras.preprocessing.image import ImageDataGenerator

#    GP=hf.ReadConfig(opts.config_file)
    batch_size = GP['batch_size']
    learning_rate = GP['learning_rate']
    kerasDefaults = p2c.keras_default_config()

##### Read Data ########
    (data_files, fields)=p2c.get_list_of_data_files(GP)

    ## Define datagenerator
    datagen=hf.ImageNoiseDataGenerator(corruption_level=GP['noise_factor'])

    ## get data dimension ##
    num_samples = 0
    for f in data_files[0:1]:
        X=np.load(f)
        num_samples += X.shape[0]

    X=np.load(data_files[0])
    print (X.shape)

    molecular_hidden_layers=GP['molecular_num_hidden']

    if not molecular_hidden_layers:
        X_train=hf.get_data(X,case=GP['case'])
        input_dim=X_train.shape[1]
    else:
        ## computing input dimension for outer AE
        input_dim=X.shape[1]*molecular_hidden_layers[-1]

    print ('The input dimension is ', input_dim)

    ## get data dimension for molecular autoencoder
    if not GP['type_bool']:
        molecular_input_dim=np.prod([X.shape[2],X.shape[3]-5])## only consider molecular location coordinates
        molecular_output_dim=np.prod([X.shape[2],X.shape[3]-5])
    else:
        molecular_input_dim=np.prod(X.shape[2:])
        molecular_output_dim=np.prod(X.shape[2:])

    print ('Data Format:\n  [Frames (%s), Molecules (%s), Beads (%s), %s (%s)]' % (
        num_samples, X.shape[1], X.shape[2], fields.keys(), X.shape[3]))

### Define Model, Solver and Compile ##########
    print ('Define the model and compile')
    opt = p2ck.build_optimizer(GP['optimizer'], learning_rate, kerasDefaults)

    print ('using mlp network')
    model_type='mlp'
    hidden_layers=GP['num_hidden']
    model=hf.dense_auto(weights_path=GP['weight_path'],input_shape=(input_dim,),nonlinearity='elu',\
                        hidden_layers=hidden_layers,l2_reg=GP['weight_decay'])
    memo='%s_%s'%(GP['base_memo'],model_type)

    print ('Autoencoder Regression problem')
    model.compile(optimizer=opt, loss='mean_squared_error')

#### Print Model Stats ###########
    KEU.Model_Info(model)
    #model.summary()

######## Define Molecular Model, Solver and Compile #########
    molecular_nonlinearity=GP['molecular_nonlinearity']

    len_molecular_hidden_layers=len(molecular_hidden_layers)
    conv_bool=GP['conv_bool']
    if conv_bool:
        molecular_model=hf.conv_dense_auto(weights_path=None,input_shape=(1,molecular_input_dim),nonlinearity=molecular_nonlinearity,\
        hidden_layers=molecular_hidden_layers,l2_reg=GP['weight_decay'])
    else:
        molecular_model=hf.dense_auto(weights_path=None,input_shape=(molecular_input_dim,),nonlinearity=molecular_nonlinearity,\
        hidden_layers=molecular_hidden_layers,l2_reg=GP['weight_decay'])

    molecular_model.compile(optimizer=opt, loss='mean_squared_error',metrics=['mean_squared_error'])
    molecular_model.summary()
    ##### set up callbacks and cooling for the molecular_model ##########
    drop=0.5
    mb_epochs = GP['molecular_epochs']
    initial_lrate=GP['learning_rate']
    epochs_drop=1+int(np.floor(mb_epochs/3))
    def step_decay(epoch):
        global initial_lrate,epochs_drop,drop
        lrate = initial_lrate * np.power(drop, np.floor((1+epoch)/epochs_drop))
        return lrate
    lr_scheduler = LearningRateScheduler(step_decay)
    history = callbacks.History()
    #callbacks=[history,lr_scheduler]
    GP.update(compute_trainable_params(model))
    candleRemoteMonitor = CandleRemoteMonitor(params=GP)
    timeoutMonitor = TerminateOnTimeOut(TIMEOUT)
    callbacks=[history, candleRemoteMonitor,timeoutMonitor]
    loss = 0.

#### Train the Model
    if GP['train_bool']:
        if not str2bool(GP['cool']):
            effec_epochs=GP['epochs']
            ct=hf.Candle_Composite_Train(datagen, model, molecular_model, data_files,mb_epochs,effec_epochs,callbacks,batch_size=32, case=GP['case'],scale_factor=0.5,len_molecular_hidden_layers=len_molecular_hidden_layers,conv_bool=conv_bool,type_bool=GP['type_bool'])
#            ct=hf.Candle_Train(datagen,model,data_files,effec_epochs,case=GP['case'])
            loss=ct.train_ac()
        else:
            effec_epochs=GP['epochs']//3
            ct=hf.Candle_Train(datagen,model,data_files,effec_epochs,case=GP['case'])
            loss=[]
            for i in range(3):
                lr=GP['learning_rate']/10**i
                ct.model.optimizer.lr.set_value(lr)
                if i>0:
                    ct.print_data=False
                    print ('Cooling Learning Rate by factor of 10...')
                loss.extend(ct.train_ac())

        if False and GP['save_path']!=None:
            if not os.path.exists(GP['save_path']):
                os.makedirs(GP['save_path'])

            loss_file='%s/%s.pkl'%(GP['save_path'],memo)
            model_file='%s/%s.hdf5'%(GP['save_path'],memo)
            o=open(loss_file,'wb')
            pickle.dump(loss,o)
            o.close()
            model.save_weights(model_file)

    return loss

def main():

    gParameters = initialize_parameters()
    run(gParameters)

if __name__ == '__main__':
    main()
    try:
        K.clear_session()
    except AttributeError:      # theano does not have this function
        pass
