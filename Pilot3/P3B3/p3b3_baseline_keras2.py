from __future__ import print_function
import numpy as np
import os, sys, gzip
import urllib, zipfile
TIMEOUT=1800 # in sec; set this to -1 for no timeout

import keras
from keras import backend as K
import math
from keras.layers.core import Dense, Dropout
from keras import optimizers

from keras.layers import Input
from keras.models import Model

from sklearn.metrics import f1_score

import argparse

import p3b3
import p3_common as p3c
import p3_common_keras as p3ck
from solr_keras import CandleRemoteMonitor, compute_trainable_params, TerminateOnTimeOut

import keras_mt_shared_cnn



def get_p3b3_parser():
        parser = argparse.ArgumentParser(prog='p3b3_baseline',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            description='Multi-task CNN for data extraction from clinical reports - Pilot 3 Benchmark 3')

        return p3b3.common_parser(parser)

def initialize_parameters():
    parser = get_p3b3_parser()
    args = parser.parse_args()
    print('Args', args)

    GP=p3b3.read_config_file(args.config_file)
    print(GP)

    GP = p3c.args_overwrite_config(args, GP)
    return GP



def run_cnn( GP, train_x, train_y, test_x, test_y,
    learning_rate = 0.01,
    batch_size = 10,
    epochs = 10,
    dropout = 0.5,
    optimizer = 'adam',
    wv_len = 300,
    filter_sizes = [3,4,5],
    num_filters = [300,300,300],
    emb_l2 = 0.001,
    w_l2 = 0.01
    ):

    max_vocab = np.max( train_x )
    max_vocab2 = np.max( test_x )
    if max_vocab2 > max_vocab:
        max_vocab = max_vocab2

    wv_mat = np.random.randn( max_vocab + 1, wv_len ).astype( 'float32' ) * 0.1

    num_classes = []
    num_classes.append( np.max( train_y[ :, 0 ] ) + 1 )
    num_classes.append( np.max( train_y[ :, 1 ] ) + 1 )
    num_classes.append( np.max( train_y[ :, 2 ] ) + 1 )
    num_classes.append( np.max( train_y[ :, 3 ] ) + 1 )


    kerasDefaults = p3c.keras_default_config()
    optimizer = p3ck.build_optimizer( optimizer, learning_rate, kerasDefaults )


    cnn = keras_mt_shared_cnn.init_export_network(
        num_classes= num_classes,
        in_seq_len= 1500,
        vocab_size= len( wv_mat ),
        wv_space= wv_len,
        filter_sizes= filter_sizes,
        num_filters= num_filters,
        concat_dropout_prob = dropout,
        emb_l2= emb_l2,
        w_l2= w_l2,
        optimizer= optimizer )

    print( cnn.summary() )

    validation_data = ( { 'Input': test_x },
        { 'Dense0': test_y[ :, 0 ],
          'Dense1': test_y[ :, 1 ],
          'Dense2': test_y[ :, 2 ],
          'Dense3': test_y[ :, 3 ] } )

    candleRemoteMonitor = CandleRemoteMonitor(params= GP)
    timeoutMonitor = TerminateOnTimeOut(TIMEOUT)

    history = cnn.fit(
        x= np.array( train_x ),
        y= [ np.array( train_y[ :, 0 ] ),
             np.array( train_y[ :, 1 ] ),
             np.array( train_y[ :, 2 ] ),
             np.array( train_y[ :, 3 ] ) ],
        batch_size= batch_size,
        epochs= epochs,
        verbose= 2,
        validation_data= validation_data,
        callbacks = [candleRemoteMonitor, timeoutMonitor]
     )

    return history


def run( GP ):
    filter_sizes = []
    num_filters = []

    start = GP[ 'filter_sizes' ]
    end = start + GP[ 'filter_sets' ] 
    n_filters = GP[ 'num_filters' ]
    for k in range( start, end ):
        filter_sizes.append( k )
        num_filters.append( n_filters )

    learning_rate = GP[ 'learning_rate' ]
    batch_size = GP[ 'batch_size' ]
    epochs = GP[ 'epochs' ]
    dropout = GP[ 'dropout' ]
    optimizer = GP[ 'optimizer' ]

    wv_len = GP[ 'wv_len' ]
    emb_l2 = GP[ 'emb_l2' ]
    w_l2 = GP[ 'w_l2' ]

    
    '''
    ## Read files
    file_path = os.path.dirname(os.path.realpath(__file__))
    print(file_path)
    lib_path = os.path.abspath(os.path.join(file_path, '..', '..', 'common'))
    sys.path.append(lib_path)

    from data_utils import get_file
    origin = 'http://ftp.mcs.anl.gov/pub/candle/public/benchmarks/P3B1/P3B1_data.tar.gz'
    data_set = 'P3B1_data'
    data_path = get_file(data_set, origin, untar=True, md5_hash=None, cache_subdir='P3B1')

    print('Data downloaded and stored at: ' + os.path.dirname(data_path))
    print('Data path:' + data_path)
    '''
    data_path = '/lustre/atlas/proj-shared/csc249/yoonh/Benchmarks/Data/Pilot3'

    train_x = np.load( data_path + '/train_X.npy' )
    train_y = np.load( data_path + '/train_Y.npy' )
    test_x = np.load( data_path + '/test_X.npy' )
    test_y = np.load( data_path + '/test_Y.npy' )


    ret = run_cnn(
        GP, 
        train_x, train_y, test_x, test_y,
        learning_rate = learning_rate,
        batch_size = batch_size,
        epochs = epochs,
        dropout = dropout,
        optimizer = optimizer,
        wv_len = wv_len,
        filter_sizes = filter_sizes,
        num_filters = num_filters,
        emb_l2 = emb_l2,
        w_l2 = w_l2       
    )

    print( 'Average loss:', str( ret.history['val_loss'] ) )
    return ret


if __name__  == "__main__":
    gParameters=initialize_parameters()
    avg_loss = run(gParameters)

