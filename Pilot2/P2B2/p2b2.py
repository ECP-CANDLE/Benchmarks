from __future__ import absolute_import
from __future__ import print_function
import theano
import matplotlib
if 'MACOSX' in matplotlib.get_backend().upper():
  matplotlib.use('TKAgg')
import pylab as py
py.ion() ## Turn on plot visualization

import gzip,pickle
import numpy as np
from PIL import Image
import cv2
import keras.backend as K
K.set_image_dim_ordering('th')
from keras.layers import Input, merge, TimeDistributed,LSTM,GRU,RepeatVector
from keras.models import Sequential,Model
from keras.layers.core import Flatten, Dense, Dropout, Activation, Reshape
from keras.initializers import normal, identity, he_normal,glorot_normal,glorot_uniform,he_uniform
from keras.layers.normalization import BatchNormalization
import threading
try:
    import configparser
except ImportError:
    import ConfigParser as configparser
from tqdm import *
import re,copy
import os
import sys

file_path = os.path.dirname(os.path.realpath(__file__))
lib_path = os.path.abspath(os.path.join(file_path, '..', 'common'))
sys.path.append(lib_path)
lib_path2 = os.path.abspath(os.path.join(file_path, '..', '..', 'common'))
sys.path.append(lib_path2)

import p2_common

def common_parser(parser):

    parser.add_argument("--config_file", dest='config_file', type=str,
                        default=os.path.join(file_path, 'p2b2_default_model.txt'),
                        help="specify model configuration file")

    # Parse has been split between arguments that are common with the default neon parser
    # and all the other options
    parser = p2_common.get_default_neon_parse(parser)
    parser = p2_common.get_p2_common_parser(parser)

    # Arguments that are applicable just to p2b2
    parser = p2b2_parser(parser)

    return parser

def p2b2_parser(parser):
    ### Hyperparameters and model save path

#    parser.add_argument("--train", action="store_true",dest="train_bool",default=True,help="Invoke training")
#    parser.add_argument("--evaluate", action="store_true",dest="eval_bool",default=False,help="Use model for inference")
#    parser.add_argument("--home-dir",help="Home Directory",dest="home_dir",type=str,default='.')
    parser.add_argument("--save-dir",help="Save Directory",dest="save_path",type=str,default=None)
    parser.add_argument("--config-file",help="Config File",dest="config_file",type=str,default=os.path.join(file_path, 'p2b2_small_model.txt'))
    parser.add_argument("--model-file",help="Trained Model Pickle File",dest="weight_path",type=str,default=None)
    parser.add_argument("--memo",help="Memo",dest="base_memo",type=str,default=None)
    parser.add_argument("--seed", action="store_true",dest="seed",default=False,help="Random Seed")
    parser.add_argument("--case",help="[Full, Center, CenterZ]",dest="case",type=str,default='CenterZ')
    parser.add_argument("--fig", action="store_true",dest="fig_bool",default=False,help="Generate Prediction Figure")
    parser.add_argument("--data-set",help="[3k_Disordered, 3k_Ordered, 3k_Ordered_and_gel, 6k_Disordered, 6k_Ordered, 6k_Ordered_and_gel]",dest="set_sel",
		type=str,default="3k_Disordered")
    #(opts,args)=parser.parse_args()
    return parser


def read_config_file(File):
    config=configparser.ConfigParser()
    config.read(File)
    section=config.sections()
    Global_Params={}

    Global_Params['num_hidden']    =eval(config.get(section[0],'num_hidden'))
    Global_Params['num_recurrent'] =eval(config.get(section[0],'num_recurrent'))
    Global_Params['look_back']     =eval(config.get(section[0],'look_back'))
    Global_Params['look_forward']  =eval(config.get(section[0],'look_forward'))
    Global_Params['batch_size']    =eval(config.get(section[0],'batch_size'))
    Global_Params['learning_rate'] =eval(config.get(section[0],'learning_rate'))
    Global_Params['epochs']        =eval(config.get(section[0],'epochs'))
    Global_Params['weight_decay']  =eval(config.get(section[0],'weight_decay'))
    Global_Params['noise_factor']  =eval(config.get(section[0],'noise_factor'))
    Global_Params['optimizer']     =eval(config.get(section[0],'optimizer'))
    Global_Params['loss']          =eval(config.get(section[0],'loss'))
    Global_Params['activation']    =eval(config.get(section[0],'activation'))
    # note 'cool' is a boolean
    Global_Params['cool']          =config.get(section[0],'cool')
    return Global_Params


############# Define Data Generators ################
class ImageNoiseDataGenerator(object):
    '''Generate minibatches with
    realtime data augmentation.
    '''
    def __init__(self,corruption_level=0.5):

        self.__dict__.update(locals())
        self.p=corruption_level
        self.lock = threading.Lock()

    def _flow_index(self, N, batch_size=32, shuffle=False, seed=None):
        b = 0
        total_b = 0
        while 1:
            if b == 0:
                if seed is not None:
                    np.random.seed(seed + total_b)

                if shuffle:
                    index_array = np.random.permutation(N)
                else:
                    index_array = np.arange(N)

            current_index = (b * batch_size) % N
            if N >= current_index + batch_size:
                current_batch_size = batch_size
            else:
                current_batch_size = N - current_index

            if current_batch_size == batch_size:
                b += 1
            else:
                b = 0
            total_b += 1
            yield index_array[current_index: current_index + current_batch_size], current_index, current_batch_size

    def flow(self, X, y, batch_size=32, shuffle=False, seed=None):
        assert len(X) == len(y)
        self.X = X
        self.y = y
        self.flow_generator = self._flow_index(X.shape[0], batch_size, shuffle, seed)
        return self

    def __iter__(self):
        # needed if we want to do something like for x,y in data_gen.flow(...):
        return self

    def next(self):
        # for python 2.x
        # Keep under lock only the mechainsem which advance the indexing of each batch
        # see # http://anandology.com/blog/using-iterators-and-generators/
        with self.lock:
            index_array, current_index, current_batch_size = next(self.flow_generator)
        # The transformation of images is not under thread lock so it can be done in parallel
        bX = np.zeros(tuple([current_batch_size] + list(self.X.shape)[1:]))
        for i, j in enumerate(index_array):
            x = self.X[j]
            x = self.insertnoise(x,corruption_level=self.p)
            bX[i] = x
        bY = self.y[index_array]
        return bX, bY

    def __next__(self):
        # for python 3.x
        return self.next()

    def insertnoise(self,x,corruption_level=0.5):
        return np.random.binomial(1,1-corruption_level,x.shape)*x


### Generate RNN compatible dataset
def create_dataset(dataset, look_back=1,look_forward=1):
    ## input is np.array of dim T,D
    #output is np.array X: N,look_back,D and Y: N,D
    # where N=T-look_back-1
    assert(look_back>=look_forward)
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-(look_forward-1)-1):
        a = dataset[i:(i+look_back), :]
        dataX.append(a)
        dataY.append(dataset[i + look_back:i+look_back+look_forward, :])
    dataX=np.array(dataX)
    dataY=np.array(dataY)
    if look_back-look_forward>0:
        dataY_mod=np.zeros((dataY.shape[0],dataX.shape[1],dataY.shape[2]))
        dataY_mod[:,0:dataY.shape[1],:]=dataY
    else:
        dataY_mod=dataY
    return dataX, dataY_mod

def generate_timedistributed_forecast(model,x,prediction_length=10):
    ## to be used when rnn is used for sequence to sequence mapping
    N,T,D=x.shape
    x_data=x[0,:,:].copy()
    x_revise=x.copy()
    for i in range(prediction_length):
        y_pred=model.predict(x_revise[0:1,:,:],batch_size=1)
        yf=y_pred[:,0,:]
        #print('prediction:',yf)
        x_data=np.vstack([x_data,yf])

        #x=x.reshape(T,D) ## assume N=1 ... i.e. one sample
        #print('data before prediction:\n',x)
        x_revise[0,0:T-1,:]=x_revise[0,1:T,:]
        x_revise[0,T-1:T,:]=yf
        #x=x.reshape(1,T,D)
        #print('data after appending prediction\n',x)
    return x_data

##### Define Neural Network Models ###################
def simple_test_rnn(T=1,D=1):
    input_shape=(T,D)
    input_img = Input(shape=input_shape)
    encoder=TimeDistributed(Dense(20,activation='relu'))(input_img)
    rnn=LSTM(10,activation='elu',return_sequences=True, stateful=False)(encoder)
    decoder=TimeDistributed(Dense(20,activation='relu'))(rnn)
    model=Model(outputs=decoder,inputs=input_img)
    return model


def dense_auto(weights_path=None,input_shape=(784,),hidden_layers=None,nonlinearity='relu'):
    input_img = Input(shape=input_shape)

    if hidden_layers!=None:
        if type(hidden_layers)!=list:
            hidden_layers=list(hidden_layers)
        for i,l in enumerate(hidden_layers):
            if i==0:
                encoded=Dense(l,activation=nonlinearity)(input_img)
            else:
                encoded=Dense(l,activation=nonlinearity)(encoded)

        for i,l in reversed(list(enumerate(hidden_layers))):
            if i <len(hidden_layers)-1:
                if i==len(hidden_layers)-2:
                    decoded=Dense(l,activation=nonlinearity)(encoded)
                else:
                    decoded=Dense(l,activation=nonlinearity)(decoded)
        decoded=Dense(input_shape[0])(decoded)
    else:
        decoded=Dense(input_shape[0])(input_img)

    model=Model(outputs=decoded,inputs=input_img)

    if weights_path:
        print('Loading Model')
        model.load_weights(weights_path)
    return model

def rnn_dense_auto(weights_path=None,T=1,D=1,nonlinearity='relu',hidden_layers=None,recurrent_layers=None):
    input_shape=(T,D)
    input_img = Input(shape=input_shape)

    if hidden_layers!=None:
        if type(hidden_layers)!=list:
            hidden_layers=list(hidden_layers)
        for i,l in enumerate(hidden_layers):
            if i==0:
                encoded=TimeDistributed(Dense(l,activation=nonlinearity))(input_img)
            else:
                encoded=TimeDistributed(Dense(l,activation=nonlinearity))(encoded)
            encoded=TimeDistributed(Dropout(0.2))(encoded)
        for i,l in enumerate(recurrent_layers):
            if i==0:
                rnn=GRU(l,return_sequences=True, stateful=False)(encoded)
            else:
               rnn=GRU(l,return_sequences=True, stateful=False)(rnn)
            encoded=TimeDistributed(Dropout(0.2))(encoded)

        for i,l in reversed(list(enumerate(hidden_layers))):
            if i <len(hidden_layers):
                if i==len(hidden_layers)-1:
                    decoded=TimeDistributed(Dense(l,activation=nonlinearity))(rnn)
                else:
                    decoded=TimeDistributed(Dense(l,activation=nonlinearity))(decoded)
        decoded=TimeDistributed(Dense(D))(decoded)
    else:
        decoded=TimeDistributed(Dense(D))(input_img)

    model=Model(outputs=decoded,inputs=input_img)

    if weights_path:
        print('Loading Model')
        model.load_weights(weights_path)
    return model


def get_data(X,case='Full'):
    if case.upper()=='FULL':
        X_train=X.copy().reshape(X.shape[0],np.prod(X.shape[1:]))
    if case.upper()=='CENTER':
        X_train=X.mean(axis=2).reshape(X.shape[0],np.prod(X.mean(axis=2).shape[1:]))
    if case.upper()=='CENTERZ':
        X_train=X.mean(axis=2)[:,:,2].reshape(X.shape[0],np.prod(X.mean(axis=2)[:,:,2].shape[1:]))
    return X_train

class Candle_Train():
    def __init__(self,
            datagen,model,
            numpylist,
            nb_epochs,
            case='CenterZ',
            batch_size=32,
            print_data=True,
            look_back=10,
            look_forward=1):
        self.datagen=datagen
        self.numpylist=numpylist
        self.epochs=nb_epochs
        self.case=case
        self.batch_size=batch_size
        self.model=model
        self.look_back=look_back
        self.look_forward=look_forward
        self.print_data=print_data

    def train_ac(self):
        bool_sample=False
        epoch_loss=[]
        for e in range(self.epochs):
            file_loss=[]
            for f in self.numpylist:
                if self.print_data:
                    if e==0:
                        print(f)
                X=np.load(f)
                data=get_data(X,self.case)
                X_train,y_train=create_dataset(data,look_back=self.look_back,look_forward=self.look_forward)
                imggen=self.datagen.flow(X_train, y_train, batch_size=self.batch_size)
                N_iter=X.shape[0]//self.batch_size

                iter_loss=[]
                for _ in range(N_iter+1):
                    x,y=next(imggen)
                    subset_sample_weight=np.ones((x.shape[0],1))
                    sample_weight=np.zeros((x.shape[0],self.look_back))
                    sample_weight[:,0:self.look_forward]=subset_sample_weight
                    loss_data=self.model.train_on_batch(x,y,sample_weight=sample_weight)
                    iter_loss.append(loss_data)
                file_loss.append(np.array(iter_loss).mean(axis=0))
            print('\nLoss on epoch %d:'%e, file_loss[-1])
            epoch_loss.append(np.array(file_loss).mean(axis=0))
        return epoch_loss


class autoencoder_preprocess():
    def __init__(self,img_size=(784,),noise_factor=0.):
        self.noise=noise_factor
        self.img_size=img_size
        self.lock = threading.Lock()

    def add_noise(self,X_train):
        ## Add noise to input data
        np.random.seed(100)
        ind=np.where(X_train==0)
        rn=self.noise*np.random.rand(np.shape(ind)[1])
        X_train[ind]=rn
        return X_train

    def renormalize(self,X_train,mu,sigma):
        X_train=(X_train-mu)/sigma
        X_train = X_train.astype("float32")
        return X_train

def get_activations(model, layer, X_batch):
    get_activations = K.function([model.layers[0].input, K.learning_phase()], model.layers[layer].output)
    activations = get_activations([X_batch,0])
    return activations
