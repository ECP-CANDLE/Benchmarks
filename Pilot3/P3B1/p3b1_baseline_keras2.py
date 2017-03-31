import numpy as np
import os, sys, gzip
import urllib, zipfile
from MTL_run import run_mtl

from sklearn.metrics import f1_score

def do_10_fold():
    shared_nnet_spec= [ 1200 ]
    individual_nnet_spec0= [ 1200, 1200 ]
    individual_nnet_spec1= [ 1200, 1200 ]
    individual_nnet_spec2= [ 1200, 1200 ]
    individual_nnet_spec = [ individual_nnet_spec0, individual_nnet_spec1, individual_nnet_spec2 ]

    learning_rate = 0.01
    batch_size = 10
    n_epochs = 10
    dropout = 0.0


    truth0 = []
    pred0 = []

    truth1 = []
    pred1 = []

    truth2 = []
    pred2 = []


   ## Read files
    file_path = os.path.dirname(os.path.realpath(__file__))
    print file_path
    lib_path = os.path.abspath(os.path.join(file_path, '..', '..', 'common'))
    sys.path.append(lib_path)

    from data_utils import get_file
    origin = 'http://ftp.mcs.anl.gov/pub/candle/public/benchmarks/P3B1/P3B1_data.tgz'
    data_loc = get_file('P3B1_data.tgz', origin, untar=True, md5_hash=None, cache_subdir='P3B1')

    print 'Data downloaded and stored at: ' + data_loc
    data_path = os.path.dirname(data_loc)
    print data_path

    for fold in range( 1 ):

        feature_train_0 = np.genfromtxt( data_path + '/task0_' + str( fold ) + '_train_feature.csv', delimiter= ',' )
        truth_train_0 = np.genfromtxt( data_path + '/task0_' + str( fold ) + '_train_label.csv', delimiter= ',' )
        feature_test_0 = np.genfromtxt( data_path + '/task0_' + str( fold ) + '_test_feature.csv', delimiter= ',' )
        truth_test_0 = np.genfromtxt( data_path + '/task0_' + str( fold ) + '_test_label.csv', delimiter= ',' )

        feature_train_1 = np.genfromtxt( data_path + '/task1_' + str( fold ) + '_train_feature.csv', delimiter= ',' )
        truth_train_1 = np.genfromtxt( data_path + '/task1_' + str( fold ) + '_train_label.csv', delimiter= ',' )
        feature_test_1 = np.genfromtxt( data_path + '/task1_' + str( fold ) + '_test_feature.csv', delimiter= ',' )
        truth_test_1 = np.genfromtxt( data_path + '/task1_' + str( fold ) + '_test_label.csv', delimiter= ',' )

        feature_train_2 = np.genfromtxt( data_path + '/task2_' + str( fold ) + '_train_feature.csv', delimiter= ',' )
        truth_train_2 = np.genfromtxt( data_path + '/task2_' + str( fold ) + '_train_label.csv', delimiter= ',' )
        feature_test_2 = np.genfromtxt( data_path + '/task2_' + str( fold ) + '_test_feature.csv', delimiter= ',' )
        truth_test_2 = np.genfromtxt( data_path + '/task2_' + str( fold ) + '_test_label.csv', delimiter= ',' )

        features_train = [ feature_train_0, feature_train_1, feature_train_2 ]
        truths_train = [ truth_train_0, truth_train_1, truth_train_2 ]
        features_test = [ feature_test_0, feature_test_1, feature_test_2 ]
        truths_test = [ truth_test_0, truth_test_1, truth_test_2 ]


        ret = run_mtl(
            features_train= features_train,
            truths_train= truths_train,
            features_test= features_test,
            truths_test= truths_test,
            shared_nnet_spec= shared_nnet_spec,
            individual_nnet_spec= individual_nnet_spec,
            learning_rate= learning_rate,
            batch_size= batch_size,
            n_epochs= n_epochs,
            dropout= dropout
        )

        truth0.extend( ret[ 0 ][ 0 ] )
        pred0.extend( ret[ 0 ][ 1 ] )

        truth1.extend( ret[ 1 ][ 0 ] )
        pred1.extend( ret[ 1 ][ 1 ] )

        truth2.extend( ret[ 2 ][ 0 ] )
        pred2.extend( ret[ 2 ][ 1 ] )


    print 'Task 1: Primary site - Macro F1 score', f1_score( truth0, pred0, average= 'macro' )
    print 'Task 1: Primary site - Micro F1 score', f1_score( truth0, pred0, average= 'micro' )

    print 'Task 2: Tumor laterality - Macro F1 score', f1_score( truth1, pred1, average= 'macro' )
    print 'Task 3: Tumor laterality - Micro F1 score', f1_score( truth1, pred1, average= 'micro' )

    print 'Task 3: Histological grade - Macro F1 score', f1_score( truth2, pred2, average= 'macro' )
    print 'Task 3: Histological grade - Micro F1 score', f1_score( truth2, pred2, average= 'micro' )



if __name__  == "__main__":
    do_10_fold()
