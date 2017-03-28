import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.optimizers import RMSprop
import numpy as np
import os

import datetime
import cPickle



class LossHistory( keras.callbacks.Callback ):
    def on_train_begin( self, logs= {} ):
        self.losses = []

    def on_batch_end( self, batch, logs= {} ):
        self.losses.append( logs.get( 'loss' ) )


rnn_size = 256


# load data from pickle
f = open( 'data.pkl', 'r' )

classes = cPickle.load( f )
chars = cPickle.load( f )
char_indices = cPickle.load( f )
indices_char = cPickle.load( f )

maxlen = cPickle.load( f )
step = cPickle.load( f )

X_ind = cPickle.load( f )
y_ind = cPickle.load( f )

f.close()

[ s1, s2 ] = X_ind.shape

X = np.zeros( ( s1, s2, len( chars ) ), dtype=np.bool )
y = np.zeros( ( s1, len( chars ) ), dtype=np.bool )

for i in range( s1 ):
    for t in range( s2 ):
        X[ i, t, X_ind[ i, t ] ] = 1
    y[ i, y_ind[ i ] ] = 1

# build the model: a single LSTM
print( 'Build model...' )
model = Sequential()
model.add( LSTM( rnn_size, input_shape=( maxlen, len( chars ) ) ) )
model.add( Dense( len( chars ) ) )
model.add( Activation( 'softmax' ) )

optimizer = RMSprop( lr= 0.001 )
model.compile( loss= 'categorical_crossentropy', optimizer= optimizer )


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# train the model, output generated text after each iteration
min_loss = 1e15
loss_count = 0

for iteration in range(1, 100):
    print()
    print('-' * 50)
    print('Iteration', iteration)

    history = LossHistory()
    model.fit( X, y, batch_size= 100, nb_epoch= 1, callbacks= [ history ] )

    loss = history.losses[ -1 ]
    print( loss )

    if loss < min_loss:
        min_loss = loss
        loss_count = 0
    else:
        loss_count = loss_count + 1
    if loss_count > 4:
        break

    dirname =  str( rnn_size ) + "/" + str( maxlen )
    if not os.path.exists( dirname ):
        os.makedirs( dirname )

    # serialize model to JSON
    model_json = model.to_json()
    with open( dirname + "/model_" + str( iteration ) + "." + str( round( loss, 6 ) ) + ".json", "w" ) as json_file:
        json_file.write( model_json )
    # serialize weights to HDF5
    model.save_weights( dirname + "/model_" + str( iteration ) + "." + str( round( loss, 6 ) ) + ".h5" )
    print( "Checkpoint saved." )

    outtext = open( dirname + "/example_" + str( iteration ) + "." + str( round( loss, 6 ) ) + ".txt", "w" )

    for diversity in [0.2, 0.5, 1.0, 1.2]:
        outtext.write('----- diversity:' + str( diversity ) + "\n" )

        generated = ''
        seedstr = "Diagnosis"
        outtext.write('----- Generating with seed: "' + seedstr + '"' + "\n" )

        sentence = " " * maxlen

        # class_index = 0
        generated += sentence
        outtext.write( generated )

        for c in seedstr:
            sentence = sentence[1:] + c
            x = np.zeros( ( 1, maxlen, len( chars ) ) )
            for t, char in enumerate(sentence):
                x[ 0, t, char_indices[ char ] ] = 1.

            preds = model.predict(x, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            generated += c

            outtext.write( c )


        for i in range( 400 ):
            x = np.zeros( ( 1, maxlen, len( chars ) ) )
            for t, char in enumerate(sentence):
                x[ 0, t, char_indices[ char ] ] = 1.

            preds = model.predict(x, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char

            outtext.write(next_char)

        outtext.write( "\n" )

    outtext.close()
