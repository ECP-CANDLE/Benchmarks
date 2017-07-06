import theano
import numpy as np
import scipy as sp
import pickle
import sys,os
import argparse
import matplotlib
from sklearn.preprocessing import MinMaxScaler
matplotlib.use('TKAgg')
import pylab as py
py.ion()
file_path = os.path.dirname(os.path.realpath(__file__))
lib_path = os.path.abspath(os.path.join(file_path, '..', 'common'))
sys.path.append(lib_path)
lib_path2 = os.path.abspath(os.path.join(file_path, '..','..', 'common'))
sys.path.append(lib_path2)

from keras import backend as K

from data_utils import get_file

import p2b2 as p2b2
import p2_common as p2c
import p2_common_keras as p2ck


HOME=os.environ['HOME']
def parse_list(option, opt, value, parser):
  setattr(parser.values, option.dest, value.split(','))
def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")

def get_p2b2_parser():
        parser = argparse.ArgumentParser(prog='p2b2_baseline',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            description='Train Molecular Sequence Predictor - Pilot 2 Benchmark 2')   

        return p2b2.common_parser(parser)

def initialize_parameters():

    parser = get_p2b2_parser()
    args = parser.parse_args()
    print('Args', args)

    GP=p2b2.read_config_file(args.config_file)
    print(GP)

    GP = p2c.args_overwrite_config(args, GP)
    return GP

def run(GP):

    ## set the seed
    if GP['seed']:
    	np.random.seed(7)
    else:
    	np.random.seed(np.random.randint(10000))

    ## Set paths
    if not os.path.isdir(GP['home_dir']):
    	print('Keras home directory not set')
    	sys.exit(0)
    sys.path.append(GP['home_dir'])
	
    import p2b2 as hf
    reload(hf)
    
    reload(p2ck)

    maps=hf.autoencoder_preprocess()
	
    ## Import keras modules
    from keras.optimizers import Adam
    #from keras.datasets import mnist
    #from keras.callbacks import LearningRateScheduler,ModelCheckpoint
    #from keras import callbacks
    #from keras.layers.advanced_activations import ELU
    #from keras.preprocessing.image import ImageDataGenerator

    batch_size = GP['batch_size']
    learning_rate = GP['learning_rate']
    kerasDefaults = p2c.keras_default_config()

##### Read Data ########
    data_files=p2c.get_list_of_data_files(GP)

    ## Define datagenerator
    datagen=hf.ImageNoiseDataGenerator(corruption_level=GP['noise_factor'])

    X=np.load(data_files[0])
    data=hf.get_data(X,case=GP['case'])
    X_train,y_train=hf.create_dataset(data,GP['look_back'],look_forward=GP['look_forward']) ## convert data to a sequence 
    temporal_dim=X_train.shape[1]
    input_dim=X_train.shape[2]
	
    print('X_train type and shape:', X_train.dtype, X_train.shape)
    print('X_train.min():', X_train.min())
    print('X_train.max():', X_train.max())
	
### Define Model, Solver and Compile ##########
    print('Define the model and compile')
    #opt=Adam(lr=GP['learning_rate'])
    opt = p2ck.build_optimizer(GP['optimizer'], learning_rate, kerasDefaults)

    print('using mlp network')
    model_type='mlp'
    hidden_layers=GP['num_hidden']
    if len(hidden_layers)==0:
    	hidden_layers=None
    recurrent_layers=GP['num_recurrent']
	
    ## Model is a Autoencoder-RNN network
    model=hf.rnn_dense_auto(weights_path=None,T=temporal_dim,D=input_dim,nonlinearity='relu',hidden_layers=hidden_layers,recurrent_layers=recurrent_layers)
		
    memo='%s_%s'%(GP['base_memo'],model_type)

    print('Autoencoder Regression problem')
    model.compile(optimizer=opt, loss='mean_squared_error',sample_weight_mode="temporal")
    model.summary()	## print model summary in details

#### Train the Model
    if GP['train_bool']:
    	if not str2bool(GP['cool']):
            effec_epochs=GP['epochs']
    	    ct=hf.Candle_Train(datagen,model,data_files,effec_epochs,case=GP['case'],look_back=GP['look_back'],look_forward=GP['look_forward'])
            loss=ct.train_ac()
    	else:
            effec_epochs=GP['epochs']//3
            if effec_epochs==0:
    	    	effec_epochs=1
    	    ct=hf.Candle_Train(datagen,model,data_files,effec_epochs,case=GP['case'],look_back=GP['look_back'],look_forward=GP['look_forward'])
    	    loss=[]
    	    for i in range(3):
    	    	lr=GP['learning_rate']/10**i
    	    	ct.model.optimizer.lr.set_value(lr)
    	    	if i>0:
    	    	    ct.print_data=False
    	    	    print('Cooling Learning Rate by factor of 10...')
    	    	loss.extend(ct.train_ac())

    	if GP['save_path']!=None:
    	    loss_file='%s/%s.pkl'%(GP['save_path'],memo)
    	    o=open(loss_file,'wb')
    	    pickle.dump(loss,o)
    	    o.close()

    	## Generate model forecast figure  
    	if GP['fig_bool']:
    	    x=X_train[0:1]
    	    xmod=x.reshape(x.shape[1],x.shape[2])
    	    yf=hf.generate_timedistributed_forecast(model,x,X_train.shape[0])
    	    yt=yt=y_train[:,0,:]
    	    ytn=np.vstack([xmod,yt])
    	    py.figure();py.plot(ytn.mean(axis=1))
    	    py.hold('on');py.plot(yf.mean(axis=1))



def main():

    gParameters = initialize_parameters()
    run(gParameters)

if __name__ == '__main__':
    main()
    try:
        K.clear_session()
    except AttributeError:      # theano does not have this function
        pass
	
