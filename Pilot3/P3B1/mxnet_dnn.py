import numpy as np
from numpy import genfromtxt

import mxnet as mx

import argparse
import sys
import time



DROPOUT = 0.0
ACTIVATION = 'relu'
# LOSS = 'mse'
# OPTIMIZER = 'sgd'
BATCH_SIZE = 10
N_EPOCHS = 100
LEARNING_RATE= 0.01

DEF_SHARED_NNET_SPEC = '1200'
DEF_INDIV_NNET_SPEC = '1200,1200;1200,1200;1200,1200'



def get_parser():
    parser = argparse.ArgumentParser(prog='p3b1_mxnet',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-v", "--verbose", action="store_true",
                        default= True,
                        help="increase output verbosity")

    parser.add_argument("-a", "--activation", action="store",
                        default=ACTIVATION,
                        help="keras activation function to use in inner layers: relu, tanh, sigmoid...")
    parser.add_argument("-b", "--batch_size", action="store",
                        default=BATCH_SIZE, type=int,
                        help="batch size")
    parser.add_argument("-e", "--n_epochs", action="store",
                        default=N_EPOCHS, type=int,
                        help="number of training epochs")
    # parser.add_argument("-o", "--optimizer", action="store",
    #                     default=OPTIMIZER,
    #                     help="keras optimizer to use: sgd, rmsprop, ...")
    parser.add_argument("--dropout", action="store",
                        default=DROPOUT, type=float,
                        help="ratio of dropout used in fully connected layers")
    parser.add_argument('-l', '--learning_rate', action='store',
                        default= LEARNING_RATE, type=float,
                        help='learning rate')

    parser.add_argument("--train_features", action="store",
                        default='../../Data/P3B1/task0_0_train_feature.csv;../../Data/P3B1/task1_0_train_feature.csv;../../Data/P3B1/task2_0_train_feature.csv',
                        help='training feature data filenames')
    parser.add_argument("--train_truths", action="store",
                        default='../../Data/P3B1/task0_0_train_label.csv;../../Data/P3B1/task1_0_train_label.csv;../../Data/P3B1/task2_0_train_label.csv',
                        help='training truth data filenames')

    parser.add_argument("--valid_features", action="store",
                        default='../../Data/P3B1/task0_0_test_feature.csv;../../Data/P3B1/task1_0_test_feature.csv;../../Data/P3B1/task2_0_test_feature.csv',
                        help='validation feature data filenames')
    parser.add_argument("--valid_truths", action="store",
                        default='../../Data/P3B1/task0_0_test_label.csv;../../Data/P3B1/task1_0_test_label.csv;../../Data/P3B1/task2_0_test_label.csv',
                        help='validation truth data filenames')

    parser.add_argument("--output_files", action="store",
                        default='result0_0.csv;result1_0.csv;result2_0.csv',
                        help="output filename")

    parser.add_argument("--shared_nnet_spec", action="store",
                        default=DEF_SHARED_NNET_SPEC,
                        help='network structure of shared layer')
    parser.add_argument("--individual_nnet_spec", action="store",
                        default=DEF_INDIV_NNET_SPEC,
                        help='network structore of task-specific layer')

    return parser



def run_mtl( features_train= [], truths_train= [], features_test= [], truths_test= [],
             shared_nnet_spec= [],
             individual_nnet_spec= [],
             learning_rate= 0.01,
             batch_size= 10,
             n_epochs= 10,
             dropout= 0.0,
             verbose= 1,
             activation= 'relu'
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


    train_iterators = []
    val_iterators = []

    for l in range( len( truths_train ) ):
        train_iter = mx.io.NDArrayIter( features_train[ l ], labels_train[ l ], batch_size, shuffle= True )
        val_iter = mx.io.NDArrayIter( features_test[ l ], labels_test[ l ], batch_size )

        train_iterators.append( train_iter )
        val_iterators.append( val_iter )


    data = mx.sym.Variable( 'data' )

    shared_layers = [ data ]

    for k in range( len( shared_nnet_spec ) ):
        fc = mx.sym.FullyConnected( data= shared_layers[ -1 ], name= 'fc' + str( k ), num_hidden= shared_nnet_spec[ k ] )
        shared_layers.append( fc )
        act = mx.sym.Activation( data= shared_layers[ -1 ], name= 'act' + str( k ), act_type= activation )
        shared_layers.append( act )
        if dropout > 0:
            do = mx.symbol.Dropout( data = shared_layers[ -1 ], name= 'do' + str( k ) )
            shared_layers.append( do )

    indiv_layers_arr = []
    models = []

    for l in range( len( individual_nnet_spec ) ):
        indiv_layers = []
        for k in range( len( individual_nnet_spec[ l ] ) + 1 ):
            if k == 0:
                fc = mx.sym.FullyConnected( data= shared_layers[ -1 ], name= 'fc' + str( l ) + '_' + str( k ), num_hidden= individual_nnet_spec[ l ][ k ] )
                indiv_layers.append( fc )
                act = mx.sym.Activation( data= indiv_layers[ -1 ], name= 'act' + str( l ) + '_' + str( k ), act_type= activation )
                indiv_layers.append( fc )
                if dropout > 0:
                    do = mx.symbol.Dropout( data= indiv_layers[ -1 ], name= 'do' + str( l ) + '_' + str( k ) )
            elif k < len( individual_nnet_spec[ l ] ):
                fc = mx.sym.FullyConnected( data= indiv_layers[ -1 ], name= 'fc' + str( l ) + '_' + str( k ), num_hidden= individual_nnet_spec[ l ][ k ] )
                indiv_layers.append( fc )
                act = mx.sym.Activation( data= indiv_layers[ -1 ], name= 'act' + str( l ) + '_' + str( k ), act_type= activation )
                indiv_layers.append( act )
                if dropout > 0:
                    do = mx.symbol.Dropout( data= indiv_layers[ -1 ], name= 'do' + str( l ) + '_' + str( k ) )
                    indiv_layers.append( act )
            else:
                fc = mx.sym.FullyConnected( data= indiv_layers[ -1 ], name= 'fc' + str( l ) + '_' + str( k ), num_hidden= n_out_nodes[ l ] )
                indiv_layers.append( fc )
                mlp = mx.sym.SoftmaxOutput( data= indiv_layers[ -1 ], name= 'softmax' )
                indiv_layers.append( mlp )

        indiv_layers_arr.append( indiv_layers )

        model = mx.mod.Module( symbol= indiv_layers[ -1 ] )
        models.append( model )



    metric = mx.metric.Accuracy()

    for i in range( n_epochs ):
        for k in range( len( models ) ):
            model = models[ k ]
            train_iter = train_iterators[ k ]
            val_iter = val_iterators[ k ]

            model.fit(
                train_data= train_iter,
                eval_data= val_iter,
                eval_metric= 'acc',
                optimizer= 'sgd',
                optimizer_params= { 'learning_rate': learning_rate, 'momentum': 0.0 },
                num_epoch= 1,
                eval_batch_end_callback= mx.callback.Speedometer( batch_size, 1 )
            )

            valid_acc = model.score( val_iter, eval_metric= metric )
            print( 'Task ' + str( k ) + ': ' + str( valid_acc ) )



if __name__  == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    print( 'Args:', args )

    if args.verbose:
        verbose = 1
    else:
        verbose = 0

    shared_nnet_spec = []
    elem = args.shared_nnet_spec.split( ',' )
    for el in elem:
        shared_nnet_spec.append( int( el ) )

    individual_nnet_spec = []
    indiv = args.individual_nnet_spec.split( ';' )
    for ind in indiv:
        indiv_nnet_spec = []
        elem = ind.split( ',' )
        for el in elem:
            indiv_nnet_spec.append( int( el ) )
        individual_nnet_spec.append( indiv_nnet_spec )

    features_train = []
    truths_train = []

    feature_files = args.train_features.split( ';' )
    truth_files = args.train_truths.split( ';' )

    if len( feature_files ) != len( individual_nnet_spec ) or len( truth_files ) != len( individual_nnet_spec ):
        print( "Number of network specifications and number of training datafile mismatch" )
        sys.exit()

    for i in range( len( feature_files ) ):
        feature_train_0 = np.genfromtxt( feature_files[ i ], delimiter= ',' )
        truth_train_0 = np.genfromtxt( truth_files[ i ], delimiter= ',' )
        features_train.append( feature_train_0 )
        truths_train.append( truth_train_0 )

    features_valid = []
    truths_valid = []

    feature_files = args.valid_features.split( ';' )
    truth_files = args.valid_truths.split( ';' )

    if len( feature_files ) != len( individual_nnet_spec ) or len( truth_files ) != len( individual_nnet_spec ):
        print( "Number of network specifications and number of validation datafile mismatch" )
        sys.exit()

    for i in range( len( feature_files ) ):
        feature_valid_0 = np.genfromtxt( feature_files[ i ], delimiter= ',' )
        truth_valid_0 = np.genfromtxt( truth_files[ i ], delimiter= ',' )
        features_valid.append( feature_valid_0 )
        truths_valid.append( truth_valid_0 )

    if len( args.output_files ) == 0:
        out_files = []
        for i in range( len( individual_nnet_spec ) ):
            filename = 'result' + str( i ) + '.csv'
            out_files.append( filename )
    else:
        out_files = args.output_files.split( ';' )

        if len( out_files ) != len( individual_nnet_spec ):
            print( "Number of network specifications and number of output datafile mismatch" )
            sys.exit()

    run_mtl(
            features_train= features_train,
            truths_train= truths_train,
            features_test= features_valid,
            truths_test= truths_valid,

            shared_nnet_spec= shared_nnet_spec,
            individual_nnet_spec= individual_nnet_spec,

            learning_rate= args.learning_rate,
            batch_size= args.batch_size,
            n_epochs= args.n_epochs,
            dropout= args.dropout,
            activation= args.activation,

            verbose= verbose
        )
