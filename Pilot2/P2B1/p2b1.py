from __future__ import absolute_import
from __future__ import print_function

import numpy as np

import os
import sys
#import logging
import argparse
import glob
import threading

try:
    import configparser
except ImportError:
    import ConfigParser as configparser

from tqdm import *

import keras.backend as K # This common file (used by all the frameworks) should be keras independent !

file_path = os.path.dirname(os.path.realpath(__file__))
lib_path = os.path.abspath(os.path.join(file_path, '..'))
sys.path.append(lib_path)
lib_path2 = os.path.abspath(os.path.join(file_path, '..', '..', 'common'))
sys.path.append(lib_path2)

import default_utils
import data_utils


class BenchmarkP2B1(default_utils.Benchmark):


    def parse_from_benchmark(self):

        self.parser.add_argument("--train_data", dest='train_data',
                        default=argparse.SUPPRESS, type=str,
                        choices=['3k_Disordered', '3k_Ordered', '3k_Ordered_and_gel', '6k_Disordered', '6k_Ordered', '6k_Ordered_and_gel'],
                        help="[3k_Disordered, 3k_Ordered, 3k_Ordered_and_gel, 6k_Disordered, 6k_Ordered, 6k_Ordered_and_gel]")

        self.parser.add_argument("--molecular_epochs", action="store", type= int,
                        default=argparse.SUPPRESS,
                        help= "number of training epochs for the molecular part of the model")

        self.parser.add_argument("--molecular_num_hidden", nargs='+',
                        default=argparse.SUPPRESS,
                        help="number of units in fully connected layers for the molecular part of the model in an integer array")
    
        self.parser.add_argument("--molecular_activation",
                        default=argparse.SUPPRESS,
                        help="keras activation function to use for the molecular part of the model in inner layers: relu, tanh, sigmoid...")
        self.parser.add_argument("--conv-AE", action="store_true",dest="conv_bool",default=True,help="Invoke training using Conv1D NN for inner AE")

        self.parser.add_argument("--cool", action="store_true",
                        default=argparse.SUPPRESS,
                        help= "flag for cooling")
    
        self.parser.add_argument("--weight_decay", action="store", type=float,
                        default=argparse.SUPPRESS,
                        help= "weight decay factor")
    
        self.parser.add_argument("--case",help="[Full, Center, CenterZ]",dest="case",type=str,default='CenterZ')
        self.parser.add_argument("--fig", action="store_true",dest="fig_bool",default=False,help="Generate Prediction Figure")
    
        self.parser.add_argument("--include-type", action="store_true",dest="type_bool",default=False,help="Include molecule type information in designing AE")
    
        self.parser.add_argument("--model-file",help="Trained Model Pickle File",dest="weight_path",type=str,default=None)



    def read_config_file(self, file):
        """Functionality to read the configue file
           specific for each benchmark.
        """

        config=configparser.ConfigParser()
        config.read(file)
        section=config.sections()
        fileParams={}
    
        fileParams['activation']=eval(config.get(section[0],'activation'))
        fileParams['batch_size']=eval(config.get(section[0],'batch_size'))
        fileParams['dense']=eval(config.get(section[0],'dense'))
        fileParams['epochs']=eval(config.get(section[0],'epochs'))
        fileParams['initialization']=eval(config.get(section[0],'initialization'))
        fileParams['learning_rate']=eval(config.get(section[0], 'learning_rate'))
        fileParams['loss']=eval(config.get(section[0],'loss'))
        fileParams['metrics'] = eval(config.get(section[0],'metrics'))
        fileParams['noise_factor'] = eval(config.get(section[0],'noise_factor'))
        fileParams['optimizer']=eval(config.get(section[0],'optimizer'))
        fileParams['scaling']=eval(config.get(section[0],'scaling'))
    
        fileParams['data_url']=eval(config.get(section[0],'data_url'))
        fileParams['train_data']=eval(config.get(section[0],'train_data'))
        fileParams['model_name']=eval(config.get(section[0],'model_name'))
#        fileParams['output_dir'] = eval(config.get(section[0], 'output_dir'))

        fileParams['molecular_epochs']=eval(config.get(section[0],'molecular_epochs'))
        fileParams['molecular_num_hidden']=eval(config.get(section[0],'molecular_num_hidden'))
        fileParams['molecular_activation']=eval(config.get(section[0],'molecular_activation'))
        fileParams['cool']=eval(config.get(section[0],'cool'))
        fileParams['weight_decay']=eval(config.get(section[0],'weight_decay'))
        

        # parse the remaining values
        for k,v in config.items(section[0]):
            if not k in fileParams:
                fileParams[k] = eval(v)
    
        return fileParams


data_sets = {
  '3k_Disordered' : ('3k_run10_10us.35fs-DPPC.10-DOPC.70-CHOL.20-f20.dir', '36ebb5bbc39e1086176133c92c29b5ce'),
#  '3k_Disordered' : ('3k_run10_10us.35fs-DPPC.10-DOPC.70-CHOL.20.dir', '3a5fc83d3de48de2f389f5f0fa5df6d2'),
  '3k_Ordered' : ('3k_run32_10us.35fs-DPPC.50-DOPC.10-CHOL.40.dir', '6de30893cecbd9c66ea433df0122b328'),
  '3k_Ordered_and_gel' : ('3k_run43_10us.35fs-DPPC.70-DOPC.10-CHOL.20.dir', '45b9a2f7deefb8d5b016b1c42f5fba71'),
  '6k_Disordered' : ('6k_run10_25us.35fs-DPPC.10-DOPC.70-CHOL.20.dir', '24e4f8d3e32569e8bdd2252f7259a65b'),
  '6k_Ordered' : ('6k_run32_25us.35fs-DPPC.50-DOPC.10-CHOL.40.dir', '0b3b39086f720f73ce52d5b07682570d'),
  '6k_Ordered_and_gel' : ('6k_run43_25us.35fs-DPPC.70-DOPC.10-CHOL.20.dir', '3b3e069a7c55a4ddf805f5b898d6b1d1')
  }

from collections import OrderedDict

def gen_data_set_dict():
    # Generating names for the data set
    names= {'x' : 0, 'y' : 1, 'z' : 2, 
            'CHOL' : 3, 'DPPC' : 4, 'DIPC' : 5, 
            'Head' : 6, 'Tail' : 7}
    for i in range(12):
        temp = 'BL'+str(i+1)
        names.update({temp : i+8})

    # dictionary sorted by value
    fields=OrderedDict(sorted(names.items(), key=lambda t: t[1]))

    return fields


def get_list_of_data_files(GP):

    print ('Reading Data...')
    ## Identify the data set selected
    data_set = data_sets[GP['train_data']][0]
    ## Get the MD5 hash for the proper data set
    data_hash=data_sets[GP['train_data']][1]
    print ('Reading Data Files... %s->%s' % (GP['train_data'], data_set))
    ## Check if the data files are in the data director, otherwise fetch from FTP
    path = GP['data_url']
    data_file = default_utils.fetch_file(path + data_set + '.tar.gz', 'Pilot2', untar=True, md5_hash=data_hash)
    data_dir = os.path.join(os.path.dirname(data_file), data_set)
    ## Make a list of all of the data files in the data set
    data_files=glob.glob('%s/*.npy'%data_dir)

    fields = gen_data_set_dict()

    return (data_files, fields)



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
                b=0
                #b=None

            #if current_index + current_batch_size==N:
            #   b=None
            total_b += 1
            yield index_array[current_index: current_index + current_batch_size], current_index, current_batch_size
            #if b==None:
            #    return

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


def get_data(X,case='Full'):
    if case.upper()=='FULL':
        X_train=X.copy().reshape(X.shape[0],np.prod(X.shape[1:]))
    if case.upper()=='CENTER':
        X_train=X.mean(axis=2).reshape(X.shape[0],np.prod(X.mean(axis=2).shape[1:]))
    if case.upper()=='CENTERZ':
        X_train=X.mean(axis=2)[:,:,2].reshape(X.shape[0],np.prod(X.mean(axis=2)[:,:,2].shape[1:]))

    return X_train


## get activations for hidden layers of the model
def get_activations(model, layer, X_batch):
    get_activations = K.function([model.layers[0].input, K.learning_phase()], [model.layers[layer].output])
    activations = get_activations([X_batch,0])
    return activations


############# Define CANDLE Functionality ################

class Candle_Train():
    def __init__(self, datagen, model, numpylist,nb_epochs,case='Full',batch_size=32,print_data=True):
        self.numpylist=numpylist
        self.epochs=nb_epochs
        self.case=case
        self.batch_size=batch_size
        self.model=model
        self.datagen=datagen
        self.print_data=print_data

    def train_ac(self):
        epoch_loss=[]
        for e in tqdm(range(self.epochs)):
            file_loss=[]
            for f in self.numpylist:
                if self.print_data:
                    if e==0:
                        print (f)
                X=np.load(f)
                X_train=get_data(X,self.case)
                y_train=X_train.copy()
                imggen=self.datagen.flow(X_train, y_train, batch_size=self.batch_size)
                N_iter=X.shape[0]//self.batch_size

                iter_loss=[]
                for _ in range(N_iter+1):
                    x,y=next(imggen)
                    loss_data=self.model.train_on_batch(x,y)
                    iter_loss.append(loss_data)
                file_loss.append(np.array(iter_loss).mean(axis=0))
            print ('\nLoss on epoch %d:'%e, file_loss[-1])
            epoch_loss.append(np.array(file_loss).mean(axis=0))
        return epoch_loss


class Candle_Composite_Train():
    def __init__(self, datagen, model, molecular_ammodel, numpylist,mnb_epochs,nb_epochs,callbacks,batch_size=32,case='Full',print_data=True,scale_factor=1,epsilon=.064,len_molecular_hidden_layers=1,conv_bool=False,type_bool=False):
        self.numpylist=numpylist
        self.molecular_model=molecular_ammodel
        self.mb_epochs=mnb_epochs
        self.epochs=nb_epochs
        self.callbacks=callbacks
        self.case=case
        self.batch_size=batch_size
        self.model=model
        self.datagen=datagen
        self.print_data=print_data
        self.scale_factor=scale_factor
        self.epsilon=epsilon
        self.len_molecular_hidden_layers=len_molecular_hidden_layers
        self.conv_net=conv_bool
        self.type_feature=type_bool
    def train_ac(self):
        epoch_loss=[]
        for e in tqdm(range(self.epochs)):
            file_loss=[]
            filelist=[d for d in self.numpylist if 'AE' not in d]
            for f in filelist[0:1]:
                if self.print_data:
                    if e==0:
                        print (f)
                X=np.load(f)
                X=X[0:20,:,:,:] # please remove it for original test 
                #print(X.shape)
                #sys.exit(0)

                # Bond lengths are in the range of 0 - 10 angstroms -- normalize it to 0 - 1
                if self.type_feature:
                    Xnorm=np.concatenate([X[:,:,:,0:3]/255.,X[:,:,:,3:8],X[:,:,:,8:]/10.],axis=3)  ## normalizing the location coordinates and bond lengths and scale type encoding
                else:
                    Xnorm=np.concatenate([X[:,:,:,0:3]/255.,X[:,:,:,8:]/10.],axis=3) ## only consider the location coordinates and bond lengths per molecule
                ### Code for sub-autoencoder for molecule feature learing
                #having some problems
                num_frames=X.shape[0]
                num_molecules=X.shape[1]
                input_feature_dim=np.prod(Xnorm.shape[2:])
                XP=[]
                for i in range(num_frames):
                    #print(i)
                    #print(num_frames)
                    if self.conv_net:
                        xt=Xnorm[i].reshape(X.shape[1],1,input_feature_dim)
                        yt=Xnorm[i].reshape(X.shape[1],input_feature_dim)
                    else:
                        xt=Xnorm[i].reshape(X.shape[1],input_feature_dim)
                        yt=xt.copy()
                    w=self.molecular_model.get_weights()
                    #print (self.molecular_model.evaluate(xt,yt,verbose=0)[0])
                    while self.molecular_model.evaluate(xt,yt,verbose=0)[0]>self.epsilon:
                        print ('[Frame %d]' % (i),'Inner AE loss..', self.molecular_model.evaluate(xt,yt,verbose=0)[0])
                        self.molecular_model.set_weights(w)
                        print(xt.shape)
                        self.molecular_model.fit(xt, yt,epochs=self.mb_epochs,callbacks=self.callbacks)
                        w=self.molecular_model.get_weights()
                    yp=get_activations(self.molecular_model,self.len_molecular_hidden_layers,xt)
                    XP.append(yp)
                XP=np.array(XP)
                fout=f.split('.npy')[0]+'_AE'+'_Include%s'%self.type_feature+'_Conv%s'%self.conv_net+'.npy'
                #if e==0:
                #    np.save(fout,XP)

                # Flatten the output of the convolutional layer into a single dimension per frame
                X_train=XP.copy().reshape(XP.shape[0],np.prod(XP.shape[1:]))
                y_train=X_train.copy()
                imggen=self.datagen.flow(X_train, y_train, batch_size=self.batch_size)
                N_iter=XP.shape[0]//self.batch_size

                iter_loss=[]
                for _ in range(N_iter+1):
                    x,y=next(imggen)
                    loss_data=self.model.train_on_batch(x,y)
                    iter_loss.append(loss_data)
                #print (iter_loss)
                file_loss.append(np.array(iter_loss).mean(axis=0))


            print ('Loss on epoch %d:'%e, file_loss[-1])
            epoch_loss.append(np.array(file_loss).mean(axis=0))
        return epoch_loss

