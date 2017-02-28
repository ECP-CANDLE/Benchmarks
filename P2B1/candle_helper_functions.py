from __future__ import absolute_import
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
from keras.layers import Input
from keras.models import Sequential,Model
from keras.layers.core import Flatten, Dense, Dropout, Activation, Reshape
from keras.layers.convolutional import Convolution2D, MaxPooling2D,Convolution1D
from keras.layers.convolutional import ZeroPadding2D,UpSampling2D,Unpooling2D,perforated_Unpooling2D,DePool2D
from keras.initializations import normal, identity, he_normal,glorot_normal,glorot_uniform,he_uniform
from keras.layers.normalization import BatchNormalization
import threading

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

##### Define Neural Network Models ###################
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

    model=Model(input=input_img,output=decoded)
    
    if weights_path:
        print('Loading Model')
        model.load_weights(weights_path)
    return model

def dense_simple(weights_path=None,input_shape=(784,),nonlinearity='relu'):
    model=Sequential()
    ## encoder
    model.add(Dense(512,input_shape=input_shape,activation=nonlinearity))
    BatchNormalization()
    model.add(Dense(256,activation=nonlinearity))
    BatchNormalization()
    model.add(Dense(128,activation=nonlinearity))
    BatchNormalization()
    model.add(Dense(64,activation=nonlinearity))
    BatchNormalization()
    model.add(Dense(32,activation=nonlinearity))
    BatchNormalization()
    model.add(Dense(16,activation=nonlinearity))
    BatchNormalization()
    ## decoder
    model.add(Dense(32))
    BatchNormalization()
    model.add(Dense(64))
    BatchNormalization()
    model.add(Dense(128))
    BatchNormalization()
    model.add(Dense(256))
    BatchNormalization()
    model.add(Dense(512))
    BatchNormalization()
    model.add(Dense(input_shape[0],activation='linear'))    
    if weights_path:
        print('Loading Model')
        model.load_weights(weights_path)
    return model


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


