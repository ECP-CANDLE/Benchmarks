from __future__ import print_function
import numpy as np
import os, sys, gzip
import urllib, zipfile
TIMEOUT=1800 # in sec; set this to -1 for no timeout

from keras import backend as K
from keras.layers.core import Dense, Dropout
#from keras.optimizers import SGD

from keras import optimizers

from keras.layers import Input
from keras.models import Model

from sklearn.metrics import f1_score

import argparse

import p3b1
import p3_common as p3c
import p3_common_keras as p3ck
from solr_keras import CandleRemoteMonitor, compute_trainable_params, TerminateOnTimeOut

def get_p3b1_parser():
        parser = argparse.ArgumentParser(prog='p3b1_baseline',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            description='Multi-task (DNN) for data extraction from clinical reports - Pilot 3 Benchmark 1')

        return p3b1.common_parser(parser)

def initialize_parameters():
    parser = get_p3b1_parser()
    args = parser.parse_args()
    print('Args', args)

    GP=p3b1.read_config_file(args.config_file)
    print(GP)

    GP = p3c.args_overwrite_config(args, GP)
    return GP

def run_mtl( features_train= [], truths_train= [], features_test= [], truths_test= [],
             shared_nnet_spec= [],
             individual_nnet_spec= [],
             learning_rate= 0.01,
             batch_size= 10,
             n_epochs= 100,
             dropout= 0.0,
             verbose= 1,
             activation= 'relu',
             out_act = 'softmax',
             loss='categorical_crossentropy',
             optimizer='sgd',
             run_id= None,
             fold= None,
             gParameters= None
             ):

    labels_train = []
    labels_test = []

    n_out_nodes = []

    for l in range( len( truths_train ) ):
        truth_train_0 = truths_train[ l ]
        truth_test_0 = truths_test[ l ]

        truth_train_0 = np.array( truth_train_0, dtype= 'int32' )
        truth_test_0 = np.array( truth_test_0, dtype= 'int32' )

        mv = int( np.max( truth_train_0 ) )
        label_train_0 = np.zeros( ( len( truth_train_0 ), mv + 1 ) )
        for i in range( len( truth_train_0 ) ):
            label_train_0[ i, truth_train_0[ i ] ] = 1
        label_test_0 = np.zeros( ( len( truth_test_0 ), mv + 1 ) )
        for i in range( len( truth_test_0 ) ):
            label_test_0[ i, truth_test_0[ i ] ] = 1

        labels_train.append( label_train_0 )
        labels_test.append( label_test_0 )

        n_out_nodes.append( mv + 1 )


    shared_layers = []


    # input layer
    layer = Input( shape= ( len( features_train[ 0 ][ 0 ] ), ), name= 'input' )
    shared_layers.append( layer )


    # shared layers
    for k in range( len( shared_nnet_spec ) ):
        layer = Dense( shared_nnet_spec[ k ], activation= activation,
                       name= 'shared_layer_' + str( k ) )( shared_layers[ -1 ] )
        shared_layers.append( layer )
        if dropout > 0:
            layer = Dropout( dropout )( shared_layers[ -1 ] )
            shared_layers.append( layer )


    # individual layers
    indiv_layers_arr= []
    models = []

    trainable_count = 0
    non_trainable_count = 0

    for l in range( len( individual_nnet_spec ) ):
        indiv_layers = [ shared_layers[ -1 ] ]
        for k in range( len( individual_nnet_spec[ l ] ) + 1 ):
            if k < len( individual_nnet_spec[ l ] ):
                layer = Dense( individual_nnet_spec[ l ][ k ], activation= activation,
                               name= 'indiv_layer_' + str( l ) + '_' + str( k ) )( indiv_layers[ -1 ] )
                indiv_layers.append( layer )
                if dropout > 0:
                    layer = Dropout( dropout )( indiv_layers[ -1 ] )
                    indiv_layers.append( layer )
            else:
                layer = Dense( n_out_nodes[ l ], activation= out_act,
                               name= 'out_' + str( l ) )( indiv_layers[ -1 ] )
                indiv_layers.append( layer )

        indiv_layers_arr.append( indiv_layers )

        model = Model( input= [ shared_layers[ 0 ] ], output= [ indiv_layers[ -1 ] ] )

        # calculate trainable/non-trainable param count for each model
        param_counts = compute_trainable_params(model)
        trainable_count += param_counts['trainable_params']
        non_trainable_count += param_counts['non_trainable_params']

        models.append( model )

    # capture total param counts
    gParameters['trainable_params'] = trainable_count
    gParameters['non_trainable_params'] = non_trainable_count
    gParameters['total_params'] = trainable_count + non_trainable_count

    kerasDefaults = p3c.keras_default_config()
    optimizer = p3ck.build_optimizer(optimizer, learning_rate, kerasDefaults)

    # DEBUG - verify
    if verbose == 1:
        for k in range( len( models ) ):
            model = models[ k ]
            print('Model:',k)
            model.summary()

    for k in range( len( models ) ):
        model = models[ k ]
        model.compile( loss= loss, optimizer= optimizer, metrics= [ 'accuracy' ] )


    # train
    for epoch in range( n_epochs ):
        for k in range( len( models ) ):
            feature_train = features_train[ k ]
            label_train = labels_train[ k ]
            feature_test = features_test[ k ]
            label_test = labels_test[ k ]
            model = models[ k ]

            gParameters['run_id'] = run_id + ".{}.{}.{}".format(fold, epoch, k)
            candleRemoteMonitor = CandleRemoteMonitor(params=gParameters)
            timeoutMonitor = TerminateOnTimeOut(TIMEOUT)

            model.fit( { 'input': feature_train }, { 'out_' + str( k ) : label_train }, epochs= 1, verbose= verbose,
                callbacks= [ candleRemoteMonitor, timeoutMonitor ],
                batch_size= batch_size, validation_data= ( feature_test, label_test ) )


    # retrieve truth-pred pair
    avg_loss = 0.0
    ret = []

    for k in range( len( models ) ):
        ret_k= []

        feature_test = features_test[ k ]
        truth_test = truths_test[ k ]
        label_test = labels_test[ k ]
        model = models[ k ]

        loss = model.evaluate( feature_test, label_test )
        avg_loss = avg_loss + loss[ 0 ]

        pred = model.predict( feature_test )

        ret_k.append( truth_test )
        ret_k.append( np.argmax( pred, axis= 1 ) )

        ret.append( ret_k )

    avg_loss = avg_loss / float( len( models ) )
    ret.append( avg_loss )

    return ret

def run(GP, verbose=1):
    shared_nnet_spec = []
    elem = GP['shared_nnet_spec'].split( ',' )
    for el in elem:
        shared_nnet_spec.append( int( el ) )

    individual_nnet_spec = []
    indiv = GP['ind_nnet_spec'].split( ':' )
    for ind in indiv:
        indiv_nnet_spec = []
        elem = ind.split( ',' )
        for el in elem:
            indiv_nnet_spec.append( int( el ) )
        individual_nnet_spec.append( indiv_nnet_spec )

    learning_rate = GP['learning_rate']
    batch_size = GP['batch_size']
    n_epochs = GP['epochs']
    dropout = GP['dropout']
    activation = GP['activation']
    out_act = GP['out_act']
    loss = GP['loss']
    n_fold = GP['n_fold']
    optimizer = GP['optimizer']

    features = []
    feat = GP['feature_names'].split(':')
    for f in feat:
        features.append(f)

    n_feat = len(feat)

    print('Feature names:')
    for i in range(n_feat):
        print(features[i])

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

    # initialize arrays for all the features
    truth_array = [[] for _ in range(n_feat)]
    pred_array = [[] for _ in range(n_feat)]
    avg_loss = 0.0

    for fold in range( n_fold ):

        features_train = []
        labels_train = []

        features_test = []
        labels_test = []

        # build feature sets to match the network topology
        for i in range( len( individual_nnet_spec) ):
            feature_train_0 = np.genfromtxt(data_path + '/task'+str(i)+'_'+str(fold)+'_train_feature.csv', delimiter= ',' )
            label_train_0 = np.genfromtxt(data_path + '/task'+str(i)+'_'+str(fold)+'_train_label.csv', delimiter= ',' )
            features_train.append( feature_train_0 )
            labels_train.append( label_train_0 )

            feature_test_0 = np.genfromtxt(data_path + '/task'+str(i)+'_'+str(fold)+'_test_feature.csv', delimiter= ',' )
            label_test_0 = np.genfromtxt(data_path + '/task'+str(i)+'_'+str(fold)+'_test_label.csv', delimiter= ',' )
            features_test.append( feature_test_0 )
            labels_test.append( label_test_0 )


        ret = run_mtl(
            features_train= features_train,
            truths_train= labels_train,
            features_test= features_test,
            truths_test= labels_test,
            shared_nnet_spec= shared_nnet_spec,
            individual_nnet_spec= individual_nnet_spec,
            learning_rate= learning_rate,
            batch_size= batch_size,
            n_epochs= n_epochs,
            dropout= dropout,
            activation = activation,
            out_act = out_act,
            loss = loss,
            optimizer = optimizer,
            run_id = GP['run_id'] if 'run_id' in GP else "RUN_default",
            fold = fold,
            gParameters = GP,
            verbose = verbose
        )

        for i in range(n_feat):
            truth_array[i].extend(ret[i][0])
            pred_array[i].extend(ret[i][1])

        avg_loss += ret[ -1 ]

    avg_loss /= float( n_fold )

    for task in range(n_feat):
        print('Task',task+1,':',features[task],'- Macro F1 score', f1_score(truth_array[task], pred_array[task], average='macro'))
        print('Task',task+1,':',features[task],'- Micro F1 score', f1_score(truth_array[task], pred_array[task], average='micro'))

    return avg_loss


if __name__  == "__main__":
    gParameters=initialize_parameters()
    avg_loss = run(gParameters)
    print( avg_loss )
