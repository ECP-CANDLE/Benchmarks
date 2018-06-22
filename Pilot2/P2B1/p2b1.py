from __future__ import absolute_import
from __future__ import print_function

# import matplotlib
# if 'MACOSX' in matplotlib.get_backend().upper():
#   matplotlib.use('TKAgg')
# import pylab as py
# py.ion() ## Turn on plot visualization
# import gzip, pickle
# from PIL import Image
# import cv2
# from tqdm import *
import multiprocessing as mp
from multiprocessing import Process, Manager
import numpy as np
import keras.backend as K
import threading
try:
    import configparser
except ImportError:
    import ConfigParser as configparser
import os
import sys

file_path = os.path.dirname(os.path.realpath(__file__))
lib_path = os.path.abspath(os.path.join(file_path, '..', 'common'))
sys.path.append(lib_path)
lib_path2 = os.path.abspath(os.path.join(file_path, '..', '..', 'common'))
sys.path.append(lib_path2)

import p2_common
import helper
import random

def common_parser(parser):

    parser.add_argument("--config_file", dest='config_file', type=str,
                        default=os.path.join(file_path, 'p2b1_default_model.txt'),
                        help="specify model configuration file")

    # Parse has been split between arguments that are common with the default neon parser
    # and all the other options
    parser = p2_common.get_default_neon_parse(parser)
    parser = p2_common.get_p2_common_parser(parser)

    # Arguments that are applicable just to p2b1
    parser = p2b1_parser(parser)

    return parser


def p2b1_parser(parser):
#    parser.add_argument("--train", action="store_true",dest="train_bool",default=True,help="Invoke training")
#    parser.add_argument("--evaluate", action="store_true",dest="eval_bool",default=False,help="Use model for inference")
#    parser.add_argument("--home-dir",help="Home Directory",dest="home_dir",type=str,default='.')
    parser.add_argument("--config-file",help="Config File",dest="config_file",type=str,default=os.path.join(file_path, 'p2b1_default_model.txt'))
    parser.add_argument("--model-file",help="Trained Model Pickle File",dest="weight_path",type=str,default=None)
    parser.add_argument("--memo",help="Memo",dest="base_memo",type=str,default=None)
    parser.add_argument("--seed", action="store_true",dest="seed",default=False,help="Random Seed")
    parser.add_argument("--case",help="[Full, Center, CenterZ]",dest="case",type=str,default='Full')
    parser.add_argument("--fig", action="store_true",dest="fig_bool",default=False,help="Generate Prediction Figure")
    parser.add_argument("--data-set",help="[3k_Disordered, 3k_Ordered, 3k_Ordered_and_gel, 6k_Disordered, 6k_Ordered, 6k_Ordered_and_gel]",dest="set_sel",
        type=str,default="3k_Disordered")
    parser.add_argument("--conv-AE", action="store_true", dest="conv_bool", default=True, help="Invoke training using 1D Convs for inner AE")
    parser.add_argument("--full-conv-AE", action="store_true", dest="full_conv_bool", default=False, help="Invoke training using fully convolutional NN for inner AE")
    parser.add_argument("--include-type", action="store_true", dest="type_bool", default=True, help="Include molecule type information in desining AE")
    parser.add_argument("--nbr-type", type=str, dest="nbr_type", default='relative', help="Defines the type of neighborhood data to use. [relative, invariant]")
    parser.add_argument("--backend", help="Keras Backend", dest="backend", type=str, default='theano')
    #(opts,args)=parser.parse_args()
    return parser


#### Read Config File
def read_config_file(File):
    config = configparser.ConfigParser()
    config.read(File)
    section = config.sections()
    Global_Params = {}

    Global_Params['num_hidden']    = eval(config.get(section[0], 'num_hidden'))
    Global_Params['batch_size']    = eval(config.get(section[0], 'batch_size'))
    Global_Params['learning_rate'] = eval(config.get(section[0], 'learning_rate'))
    Global_Params['l2_reg']        = eval(config.get(section[0], 'l2_reg'))
    Global_Params['noise_factor']  = eval(config.get(section[0], 'noise_factor'))
    Global_Params['optimizer']     = eval(config.get(section[0], 'optimizer'))
    Global_Params['loss']          = eval(config.get(section[0], 'loss'))
    Global_Params['activation']    = eval(config.get(section[0], 'activation'))
    # note 'cool' is a boolean
    Global_Params['cool']          = config.get(section[0], 'cool')

    Global_Params['epochs']       =eval(config.get(section[0],'epochs'))
    Global_Params['molecular_num_hidden']   =eval(config.get(section[0],'molecular_num_hidden'))
    Global_Params['molecular_nonlinearity'] =config.get(section[0],'molecular_nonlinearity')
    Global_Params['molecular_nbrs']         =config.get(section[0],'molecular_nbrs')
    Global_Params['drop_prob']              = config.get(section[0], 'drop_prob')
    Global_Params['l2_reg']                 = eval(config.get(section[0], 'l2_reg'))
    Global_Params['sampling_density']       = eval(config.get(section[0], 'sampling_density'))
    Global_Params['save_path']              = eval(config.get(section[0], 'save_path'))

    # parse the remaining values
    for k,v in config.items(section[0]):
        if k not in Global_Params:
            Global_Params[k] = eval(v)

    return Global_Params


# get activations for hidden layers of the model
def get_activations(model, layer, X_batch):
    get_activations = K.function([model.layers[0].input, K.learning_phase()], [model.layers[layer].output])
    activations = get_activations([X_batch, 0])
    return activations


############# Define Data Generators ################
class ImageNoiseDataGenerator(object):
    '''Generate minibatches with
    realtime data augmentation.
    '''
    def __init__(self,corruption_level=0.5):

        self.__dict__.update(locals())
        self.p = corruption_level
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
                # b=None

            # if current_index + current_batch_size==N:
            #   b=None
            total_b += 1
            yield index_array[current_index: current_index + current_batch_size], current_index, current_batch_size
            # if b==None:
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
            x = self.insertnoise(x, corruption_level=self.p)
            bX[i] = x
        bY = self.y[index_array]
        return bX, bY

    def __next__(self):
        # for python 3.x
        return self.next()

    def insertnoise(self,x,corruption_level=0.5):
        return np.random.binomial(1,1-corruption_level,x.shape)*x


class autoencoder_preprocess():
    def __init__(self, img_size=(784,), noise_factor=0.):
        self.noise = noise_factor
        self.img_size = img_size
        self.lock = threading.Lock()

    def add_noise(self, X_train):
        # Add noise to input data
        np.random.seed(100)
        ind = np.where(X_train == 0)
        rn = self.noise*np.random.rand(np.shape(ind)[1])
        X_train[ind] = rn
        return X_train

    def renormalize(self, X_train, mu, sigma):
        X_train = (X_train - mu)/sigma
        X_train = X_train.astype("float32")
        return X_train

## get activations for hidden layers of the model
def get_activations(model, layer, X_batch):
    get_activations = K.function([model.layers[0].input, K.learning_phase()], [model.layers[layer].output])
    activations = get_activations([X_batch,0])
    return activations


def datagen_helper(Xnorm, frame, xt, yt):
    if self.conv_net:
        xt = Xnorm[frame]
        if self.nbr_type == 'relative':
            xt = helper.append_nbrs_relative(xt, nbrs[frame], self.molecular_nbrs)
        elif self.nbr_type == 'invariant':
            xt = helper.append_nbrs_invariant(xt, nbrs[frame], self.molecular_nbrs)
        else:
            print ('Invalid nbr_type')
            exit()

        yt = xt.copy()
        xt = xt.reshape(xt.shape[0], 1, xt.shape[1], 1)
        if self.full_conv_net:
            yt = xt.copy()

    else:
        xt = Xnorm[frame]
        if self.nbr_type == 'relative':
            xt = helper.append_nbrs_relative(xt, nbrs[frame], self.molecular_nbrs)
        elif self.nbr_type == 'invariant':
            xt = helper.append_nbrs_invariant(xt, nbrs[frame], self.molecular_nbrs)
        else:
            print ('Invalid nbr_type')
            exit()
        yt = xt.copy()

    if not len(xt_all):
        xt_all = np.expand_dims(xt, axis=0)
        yt_all = np.expand_dims(yt, axis=0)
    else:
        xt_all = np.append(xt_all, np.expand_dims(xt, axis=0), axis=0)
        yt_all = np.append(yt_all, np.expand_dims(yt, axis=0), axis=0)

class Candle_Molecular_Train():
    def __init__(self, molecular_model, molecular_encoder, files, mb_epochs, callbacks, save_path='.', batch_size=32,
                 nbr_type='relative', len_molecular_hidden_layers=1, molecular_nbrs=0,
                 conv_bool=False, full_conv_bool=False, type_bool=False, sampling_density=1.0):
        self.files = files
        self.molecular_model = molecular_model
        self.molecular_encoder = molecular_encoder
        self.mb_epochs = mb_epochs
        self.callbacks = callbacks
        self.nbr_type = nbr_type
        self.batch_size = batch_size
        self.len_molecular_hidden_layers = len_molecular_hidden_layers
        self.molecular_nbrs = molecular_nbrs
        self.conv_net = conv_bool or full_conv_bool
        self.full_conv_net = full_conv_bool
        self.type_feature = type_bool
        self.save_path = save_path+'/'
        self.sampling_density = sampling_density

        self.test_ind = random.sample(range(len(self.files)), 1)
        self.train_ind = np.setdiff1d(range(len(self.files)), self.test_ind)


    def datagen_helper(self, Xnorm, frame, xt_all, yt_all, nbrs):
        if self.conv_net:
            xt = Xnorm[frame]
            if self.nbr_type == 'relative':
                xt = helper.append_nbrs_relative(xt, nbrs[frame], self.molecular_nbrs)
            elif self.nbr_type == 'invariant':
                xt = helper.append_nbrs_invariant(xt, nbrs[frame], self.molecular_nbrs)
            else:
                print ('Invalid nbr_type')
                exit()

            yt = xt.copy()
            xt = xt.reshape(xt.shape[0], 1, xt.shape[1], 1)
            if self.full_conv_net:
                yt = xt.copy()

        else:
            xt = Xnorm[frame]
            if self.nbr_type == 'relative':
                xt = helper.append_nbrs_relative(xt, nbrs[frame], self.molecular_nbrs)
            elif self.nbr_type == 'invariant':
                xt = helper.append_nbrs_invariant(xt, nbrs[frame], self.molecular_nbrs)
            else:
                print ('Invalid nbr_type')
                exit()
            yt = xt.copy()

        if not len(xt_all):
            xt_all.expand_dims(xt, axis=0)
            yt_all.expand_dims(yt, axis=0)
        else:
            xt_all.append(xt_all, np.expand_dims(xt, axis=0), axis=0)
            yt_all.append(yt_all, np.expand_dims(yt, axis=0), axis=0)

    def datagen(self, epoch=0, print_out=1, test=0):
        print("Cpu count", mp.cpu_count())
        files = self.files
        # order = range(13, 17) # Temporarily train on only a few files range(len(files))
        # Randomize files after first training epoch
        # if epoch:
        #    order = np.random.permutation(order)

        # choose a random sample to train on
        if not test:
            order = random.sample(list(self.train_ind), int(self.sampling_density*len(self.train_ind)))
        else:
            order = self.test_ind

        for f_ind in order:
            if print_out:
                print (files[f_ind], '\n')

            (X, nbrs, resnums) = helper.get_data_arrays(files[f_ind])

            # normalizing the location coordinates and bond lengths and scale type encoding
            # Changed the xyz normalization from 255 to 350
            if self.type_feature:
                Xnorm = np.concatenate([X[:, :, :, 0:3]/320., X[:, :, :, 3:8], X[:, :, :, 8:]/10.], axis=3)

            # only consider the location coordinates and bond lengths per molecule
            else:
                Xnorm = np.concatenate([X[:, :, :, 0:3]/320., X[:, :, :, 8:]/10.], axis=3)

            num_frames = X.shape[0]

            # these lists can be shared across porcesses. Array has to be sized,
            # and cannot be appended to. Need to find a way to deal with the shared array size
            xt_all = Manager().list()
            yt_all = Manager().list()

            #for i in range(num_frames):
            num_active_frames = random.sample(range(num_frames), int(self.sampling_density*num_frames))

            print('Datagen on the following frames', num_active_frames)

            #f = lambda x: return datagen_helper(a, b, c, d, x)

            #np.array(Pool.map(f, num_active_frames)) look into this in the morning
            processes = []
            for i in num_active_frames:

                if self.conv_net:
                    xt = Xnorm[i]
                    if self.nbr_type == 'relative':
                        xt = helper.append_nbrs_relative(xt, nbrs[i], self.molecular_nbrs)
                    elif self.nbr_type == 'invariant':
                        xt = helper.append_nbrs_invariant(xt, nbrs[i], self.molecular_nbrs)
                    else:
                        print ('Invalid nbr_type')
                        exit()

                    yt = xt.copy()
                    xt = xt.reshape(xt.shape[0], 1, xt.shape[1], 1)
                    if self.full_conv_net:
                        yt = xt.copy()

                else:
                    xt = Xnorm[i]
                    if self.nbr_type == 'relative':
                        xt = helper.append_nbrs_relative(xt, nbrs[i], self.molecular_nbrs)
                    elif self.nbr_type == 'invariant':
                        xt = helper.append_nbrs_invariant(xt, nbrs[i], self.molecular_nbrs)
                    else:
                        print ('Invalid nbr_type')
                        exit()
                    yt = xt.copy()

                if not len(xt_all):
                    xt_all = np.expand_dims(xt, axis=0)
                    yt_all = np.expand_dims(yt, axis=0)
                else:
                    xt_all = np.append(xt_all, np.expand_dims(xt, axis=0), axis=0)
                    yt_all = np.append(yt_all, np.expand_dims(yt, axis=0), axis=0)
            """
                process = Process(target=self.datagen_helper, args=(Xnorm, i, xt_all, yt_all, nbrs))
                processes.append(process)

            print("proceses:", len(processes))

            for i in range(len(processes)):
                print('Starting process', i)
                processes[i].start()

            for i in range(len(processes)):
                processes[i].join()
            """
            print('xt_all: ', len(xt_all))

            yield files[f_ind], xt_all, yt_all

        return

    def train_ac(self):

        for i in range(1, self.mb_epochs+1):
            print ("\nTraining epoch: {:d}\n".format(i))

            frame_loss = []
            frame_mse = []

            current_path = self.save_path+'epoch_'+str(i)
            if not os.path.exists(current_path):
                os.makedirs(self.save_path+'/epoch_'+str(i))

            model_weight_file = '%s/%s.hdf5' % (current_path, 'model_weights')
            encoder_weight_file = '%s/%s.hdf5' % (current_path, 'encoder_weights')

            for curr_file, xt_all, yt_all in self.datagen(i):
                #for frame in random.sample(range(len(xt_all)), int(self.sampling_density*len(xt_all))):
                for frame in range(len(xt_all)):
                    history = self.molecular_model.fit(xt_all[frame], yt_all[frame], epochs=1,
                                                       batch_size=self.batch_size, callbacks=self.callbacks[:2],
                                                       verbose=0)
                    frame_loss.append(history.history['loss'])
                    frame_mse.append(history.history['mean_squared_error'])

                    if not frame % 20 or self.sampling_density != 1.0:
                        print ("Frame: {0:d}, Current history:\nLoss: {1:3.5f}\tMSE: {2:3.5f}\n"
                               .format(frame, history.history['loss'][0], history.history['mean_squared_error'][0]))

                        # Update weights filed every few frames
                        self.molecular_model.save_weights(model_weight_file)
                        self.molecular_encoder.save_weights(encoder_weight_file)

            # save Loss and mse
            print ("Saving loss and mse after current epoch... \n")
            np.save(current_path+'/loss.npy', frame_loss)
            np.save(current_path+'/mse.npy', frame_mse)

            # Update weights file
            print ("Saving weights after current epoch... \n")
            self.molecular_model.save_weights(model_weight_file)
            self.molecular_encoder.save_weights(encoder_weight_file)

            print ("Saving latent space output for current epoch... \n")
            for curr_file, xt_all, yt_all in self.datagen(0, 0, test=1):
                XP = []
                for frame in range(len(xt_all)):
                    # get latent space activation output, +1 to incorporate the flatten layer
                    # yp = get_activations(self.molecular_model, self.len_molecular_hidden_layers + 1, xt_all[frame])
                    yp = self.molecular_encoder.predict(xt_all[frame], batch_size=self.batch_size)
                    XP.append(yp)

                XP = np.array(XP)
                fout = current_path+'/'+curr_file.split('/')[-1].split('.npz')[0]+'_AE'+'_Include%s' % self.type_feature + '_Conv%s' % self.conv_net+'.npy'
                print (fout)
                np.save(fout, XP)

        return frame_loss, frame_mse
