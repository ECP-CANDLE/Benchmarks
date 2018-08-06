import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.optimizers import RMSprop
import numpy as np
import os

import datetime
import pickle

import argparse
import sys



TRAIN_FILENAME= 'data.pkl'
RNN_SIZE = 256
N_EPOCHS = 20
N_LAYERS = 1
LEARNING_RATE= 0.01
DROPOUT = 0.0
RECURRENT_DROPOUT = 0.0

TEMPERATURE = 1.0
PRIMETEXT = 'Diagnosis'
LENGTH = 1000


def get_parser():
    parser = argparse.ArgumentParser(prog='p3b2_baseline',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-v", "--verbose", action="store_true",
                        default= True,
                        help="increase output verbosity")

    parser.add_argument("-e", "--n_epochs", action="store",
                        default=N_EPOCHS, type=int,
                        help="number of training epochs")
    parser.add_argument('-l', '--learning_rate', action='store',
                        default= LEARNING_RATE, type=float,
                        help='learning rate')
    parser.add_argument("--dropout", action="store",
                        default=DROPOUT, type=float,
                        help="ratio of dropout")
    parser.add_argument("--recurrent_dropout", action="store",
                        default=RECURRENT_DROPOUT, type=float,
                        help="ratio of recurrent dropout")

    parser.add_argument("--train_data", action="store",
                        default= TRAIN_FILENAME,
                        help='training data filename')

    parser.add_argument("--output_dir", action="store",
                        default='out',
                        help="output directory")

    parser.add_argument("--rnn_size", action="store",
                        default= RNN_SIZE, type= int,
                        help='size of LSTM internal state')
    parser.add_argument("--n_layers", action="store",
                        default= N_LAYERS,
                        help='number of layers in the LSTM')

    parser.add_argument("--do_sample", action="store_true",
                        default= True,
                        help="generate synthesized text")
    parser.add_argument("--temperature", action="store",
                        default= TEMPERATURE, type= float,
                        help="variability of text synthesis")
    parser.add_argument("--primetext", action="store",
                        default= PRIMETEXT,
                        help= 'seed string of text synthesis' )
    parser.add_argument("--length", action="store",
                        default= LENGTH, type= int,
                        help= 'length of synthesized text')

    return parser





class LossHistory( keras.callbacks.Callback ):
    def on_train_begin( self, logs= {} ):
        self.losses = []

    def on_batch_end( self, batch, logs= {} ):
        self.losses.append( logs.get( 'loss' ) )



def sample( preds, temperature= 1.0 ):
    # helper function to sample an index from a probability array
    preds = np.asarray( preds ).astype( 'float64' )
    preds = np.log( preds ) / temperature
    exp_preds = np.exp( preds )
    preds = exp_preds / np.sum( exp_preds )
    probas = np.random.multinomial( 1, preds, 1 )
    return np.argmax( probas )



def char_rnn(
             rnn_size= 256,
             n_layers= 1,
             learning_rate= 0.01,
             dropout= 0.0,
             recurrent_dropout= 0.0,
             n_epochs= 100,
             data_train= 'data.pkl',
             verbose= 1,
             savedir= '',
             do_sample= True,
             temperature= 1.0,
             primetext= 'Diagnosis',
             length= 1000
             ):

        # load data from pickle
        f = open( data_train, 'rb' )

        if ( sys.version_info > ( 3, 0 ) ):
            classes = pickle.load( f, encoding= 'latin1' )
            chars = pickle.load( f, encoding= 'latin1' )
            char_indices = pickle.load( f, encoding= 'latin1' )
            indices_char = pickle.load( f, encoding= 'latin1' )

            maxlen = pickle.load( f, encoding= 'latin1' )
            step = pickle.load( f, encoding= 'latin1' )

            X_ind = pickle.load( f, encoding= 'latin1' )
            y_ind = pickle.load( f, encoding= 'latin1' )
        else:
            classes = pickle.load( f )
            chars = pickle.load( f )
            char_indices = pickle.load( f )
            indices_char = pickle.load( f )

            maxlen = pickle.load( f )
            step = pickle.load( f )

            X_ind = pickle.load( f )
            y_ind = pickle.load( f )

        f.close()

        [ s1, s2 ] = X_ind.shape
        print( X_ind.shape )
        print( y_ind.shape )
        print( maxlen )
        print( len( chars ) )

        X = np.zeros( ( s1, s2, len( chars ) ), dtype=np.bool )
        y = np.zeros( ( s1, len( chars ) ), dtype=np.bool )

        for i in range( s1 ):
            for t in range( s2 ):
                X[ i, t, X_ind[ i, t ] ] = 1
            y[ i, y_ind[ i ] ] = 1

        # build the model: a single LSTM
        if verbose:
            print( 'Build model...' )

        model = Sequential()

        # for rnn_size in rnn_sizes:
        for k in range( n_layers ):
            if k < n_layers - 1:
                ret_seq = True
            else:
                ret_seq = False

            if k == 0:
                model.add( LSTM( rnn_size, input_shape= ( maxlen, len( chars ) ), return_sequences= ret_seq,
                                 dropout= dropout, recurrent_dropout= recurrent_dropout ) )
            else:
                model.add( LSTM( rnn_size, dropout= dropout, recurrent_dropout= recurrent_dropout, return_sequences= ret_seq ) )

        model.add( Dense( len( chars ) ) )
        model.add( Activation( 'softmax' ) )

        optimizer = RMSprop( lr= learning_rate )
        model.compile( loss= 'categorical_crossentropy', optimizer= optimizer )

        if verbose:
            model.summary()


        for iteration in range( 1, n_epochs + 1 ):
            if verbose:
                print()
                print('-' * 50)
                print('Iteration', iteration)

            history = LossHistory()
            model.fit( X, y, batch_size= 100, epochs= 1, callbacks= [ history ] )

            loss = history.losses[ -1 ]
            if verbose:
                print( loss )


            dirname = savedir
            if len( dirname ) > 0 and not dirname.endswith( '/' ):
                dirname = dirname + '/'

            if not os.path.exists( dirname ):
                os.makedirs( dirname )

            # serialize model to JSON
            model_json = model.to_json()
            with open( dirname + "/model_" + str( iteration ) + "_" + "{:f}".format( loss ) + ".json", "w" ) as json_file:
                json_file.write( model_json )

            # serialize weights to HDF5
            model.save_weights( dirname + "/model_" + str( iteration ) + "_" + "{:f}".format( loss ) + ".h5" )

            if verbose:
                print( "Checkpoint saved." )

            if do_sample:
                outtext = open( dirname + "/example_" + str( iteration ) + "_" + "{:f}".format( loss ) + ".txt", "w" )

                diversity = temperature

                outtext.write('----- diversity:' + str( diversity ) + "\n" )

                generated = ''
                seedstr = primetext

                outtext.write('----- Generating with seed: "' + seedstr + '"' + "\n" )

                sentence = " " * maxlen

                for c in seedstr:
                    sentence = sentence[1:] + c
                    x = np.zeros( ( 1, maxlen, len( chars ) ) )
                    for t, char in enumerate(sentence):
                        x[ 0, t, char_indices[ char ] ] = 1.

                    preds = model.predict( x, verbose= verbose )[ 0 ]
                    next_index = sample( preds, diversity )
                    next_char = indices_char[ next_index ]

                    generated += c


                for i in range( length ):
                    x = np.zeros( ( 1, maxlen, len( chars ) ) )
                    for t, char in enumerate( sentence ):
                        x[ 0, t, char_indices[ char ] ] = 1.

                    preds = model.predict( x, verbose= verbose )[ 0 ]
                    next_index = sample( preds, diversity )
                    next_char = indices_char[ next_index ]

                    generated += next_char
                    sentence = sentence[ 1 : ] + next_char

                if (sys.version_info > (3, 0)):
                    outtext.write( generated + '\n' )
                else:
                    outtext.write( generated.decode('utf-8').encode('utf-8') + '\n' )
                outtext.close()


if __name__  == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    print( 'Args:', args )

    if args.verbose:
        verbose = 1
    else:
        verbose = 0

    ## Read files
    file_path = os.path.dirname(os.path.realpath(__file__))
    print( file_path )
    lib_path = os.path.abspath(os.path.join(file_path, '..', '..', 'common'))
    sys.path.append(lib_path)

    from data_utils import get_file
    origin = 'http://ftp.mcs.anl.gov/pub/candle/public/benchmarks/P3B2/P3B2_data.tgz'
    data_loc = get_file('P3B2_data.tgz', origin, untar=True, md5_hash=None, cache_subdir='P3B2')

    print( 'Data downloaded and stored at: ' + data_loc )
    data_path = os.path.dirname(data_loc)
    print( data_path )

    char_rnn(
        rnn_size= args.rnn_size,
        n_layers= args.n_layers,
        learning_rate= args.learning_rate,
        dropout= args.dropout,
        recurrent_dropout= args.recurrent_dropout,
        n_epochs= args.n_epochs,
        data_train= data_path+'/data.pkl',
        verbose= args.verbose,
        savedir= args.output_dir,
        do_sample= args.do_sample,
        temperature= args.temperature,
        primetext= args.primetext,
        length= args.length
    )
