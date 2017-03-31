import numpy as np

from keras.layers.core import Dense, Dropout
from keras.optimizers import SGD

from keras.layers import Input
from keras.models import Model

from sklearn.metrics import f1_score

import argparse
import sys


DROPOUT = 0.1
ACTIVATION = 'relu'
LOSS = 'mse'
# OPTIMIZER = 'sgd'
BATCH_SIZE = 10
N_EPOCHS = 10
LEARNING_RATE= 0.01

DEF_SHARED_NNET_SPEC = '1200'
DEF_INDIV_NNET_SPEC = '1200,1200;1200,1200;1200,1200'



def get_parser():
    parser = argparse.ArgumentParser(prog='p3b1_baseline',
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
                        default='data/task0_0_train_feature.csv;data/task1_0_train_feature.csv;data/task2_0_train_feature.csv',
                        help='training feature data filenames')
    parser.add_argument("--train_truths", action="store",
                        default='data/task0_0_train_label.csv;data/task1_0_train_label.csv;data/task2_0_train_label.csv',
                        help='training truth data filenames')

    parser.add_argument("--valid_features", action="store",
                        default='data/task0_0_test_feature.csv;data/task1_0_test_feature.csv;data/task2_0_test_feature.csv',
                        help='validation feature data filenames')
    parser.add_argument("--valid_truths", action="store",
                        default='data/task0_0_test_label.csv;data/task1_0_test_label.csv;data/task2_0_test_label.csv',
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
             n_epochs= 100,
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
                layer = Dense( n_out_nodes[ l ], activation= 'softmax',
                               name= 'out_' + str( l ) )( indiv_layers[ -1 ] )
                indiv_layers.append( layer )

        indiv_layers_arr.append( indiv_layers )

        model = Model( input= [ shared_layers[ 0 ] ], output= [ indiv_layers[ -1 ] ] )

        models.append( model )


    # DEBUG - verify
    if verbose == 1:
        for k in range( len( models ) ):
            model = models[ k ]
            model.summary()

    for k in range( len( models ) ):
        model = models[ k ]
        model.compile( loss= 'categorical_crossentropy', optimizer= SGD( lr= learning_rate ), metrics= [ 'accuracy' ] )


    # train
    for epoch in range( n_epochs ):
        for k in range( len( models ) ):
            feature_train = features_train[ k ]
            label_train = labels_train[ k ]
            feature_test = features_test[ k ]
            label_test = labels_test[ k ]
            model = models[ k ]

            model.fit( { 'input': feature_train }, { 'out_' + str( k ) : label_train }, nb_epoch= 1, verbose= verbose,
                batch_size= batch_size, validation_data= ( feature_test, label_test ) )


    # retrieve truth-pred pair
    ret = []

    for k in range( len( models ) ):
        ret_k= []

        feature_test = features_test[ k ]
        truth_test = truths_test[ k ]
        model = models[ k ]

        pred = model.predict( feature_test )

        ret_k.append( truth_test )
        ret_k.append( np.argmax( pred, axis= 1 ) )

        ret.append( ret_k )


    return ret



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

    ret = run_mtl(
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

    for i in range( len( out_files ) ):
        ret_array = []
        ret_array.append( ret[ i ][ 0 ] )
        ret_array.append( ret[ i ][ 1 ] )
        ret_array = np.array( ret_array, dtype= 'int32' )

        np.savetxt( out_files[ i ], np.transpose( ret_array ), fmt= '%d', delimiter= ',' )
