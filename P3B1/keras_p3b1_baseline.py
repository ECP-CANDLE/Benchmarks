import numpy

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils

from keras.layers import Input, Embedding, merge
from keras.models import Model

import cPickle, gzip

"""
This is a baseline implementation of a multi-task learning deep neural net for processing 
clinical pathology reports. The original dataa from the pathology reports cannot be made
available online. Hence, we have pre-processed the reports so that example training/testing
sets can be generated. Contact yoonh@ornl.gov for more information for generating additional
training and testing data. 

The text comprehension code takes as input an N-gram feature vector that
consists of TF-IDF representation of the words in the pathology report. Each training set
consists of 1200 samples for training, and the testing set consists of 89 samples. The MTL 
deep network train on these samples and produce the respective macro- and micro- F1 scores.  

"""

# Set up a number of initial variables for use with baseline
NUM_TASKS = 3; # number of learning tasks (for multi-task learning)
NUM_FOLDS = 10; # number of folds for training (main cross validation loop)
NUM_EPOCH = 5; # number of epochs



truth_a_arr = []
pred_a_arr = []

truth_b_arr = []
pred_b_arr = []

truth_c_arr = []
pred_c_arr = []

for fold in range( NUM_FOLDS ):

    features_train = []
    labels_train = []
    truths_train = []

    features_test = []
    labels_test = []
    truths_test = []

    n_out = []

    for task in range( NUM_TASKS ):
        file_post = '.' + str(task) + '.' + str(fold) + '.pkl.gz'
	fname_train = 'train/train' + file_post; 
	fname_test  = 'test/test' + file_post; 

        with gzip.open( fname_train, 'rb' ) as f:
            feature_train, label_train = cPickle.load( f )

        with gzip.open( fname_test, 'rb') as f:
            feature_test, label_test = cPickle.load( f )

        features_train.append( feature_train )
        labels_train.append( label_train )

        features_test.append( feature_test )
        labels_test.append( label_test )

        mv = numpy.max( label_train )
        truth_train = numpy.zeros( ( len( label_train ), mv + 1 ) )
        for i in range( len( label_train ) ):
            truth_train[ i, label_train[ i ] ] = 1

        truths_train.append( truth_train )

        mv = numpy.max( label_test )
        truth_test = numpy.zeros( ( len( label_test ), mv + 1 ) )
        for i in range( len( label_test ) ):
            truth_test[ i, label_test[ i ] ] = 1

        truths_test.append( truth_test )

        n_out.append( mv + 1 )

    flen = len( feature_train[ 0 ] ); # input feature length is set to 400 for now based on the training examples available.

    # shared layer
    main_input = Input( shape= ( flen, ), name= 'main_input' )
    layer1 = Dense( flen, activation= 'relu', name= 'layer1' )( main_input )
    layer2 = Dense( flen, activation= 'relu', name= 'layer2' )( layer1 )

    # task 1
    layer3a = Dense( flen, activation= 'relu', name= 'layer3a' )( layer2 )
    layer4a = Dense( 256, activation= 'relu', name= 'layer4a' )( layer3a )
    layer5a = Dense( n_out[ 0 ], activation= 'softmax', name= 'layer5a' )( layer4a )

    # task 2
    layer3b = Dense( flen, activation= 'relu', name= 'layer3b' )( layer2 )
    layer4b = Dense( 256, activation= 'relu', name= 'layer4b' )( layer3b )
    layer5b = Dense( n_out[ 1 ], activation= 'softmax', name= 'layer5b' )( layer4b )

    # task 3
    layer3c = Dense( flen, activation= 'relu', name= 'layer3c' )( layer2 )
    layer4c = Dense( 256, activation= 'relu', name= 'layer4c' )( layer3c )
    layer5c = Dense( n_out[ 2 ], activation= 'softmax', name= 'layer5c' )( layer4c )

    model_a = Model( input= [ main_input ], output= [ layer5a ] )
    model_b = Model( input= [ main_input ], output= [ layer5b ] )
    model_c = Model( input= [ main_input ], output= [ layer5c ] )

    model_a.summary()
    model_b.summary()
    model_c.summary()

    model_a.compile( loss= 'categorical_crossentropy', optimizer= RMSprop( lr= 0.001 ), metrics= [ 'accuracy' ] )
    model_b.compile( loss= 'categorical_crossentropy', optimizer= RMSprop( lr= 0.001 ), metrics= [ 'accuracy' ] )
    model_c.compile( loss= 'categorical_crossentropy', optimizer= RMSprop( lr= 0.001 ), metrics= [ 'accuracy' ] )

    for epoch in range( NUM_EPOCH ):
        model_a.fit( { 'main_input': features_train[ 0 ] }, { 'layer5a': truths_train[ 0 ] }, nb_epoch= 1, verbose= 1,
                    batch_size= 10, validation_data= ( features_test[ 0 ], truths_test[ 0 ] ) )

        model_b.fit( { 'main_input': features_train[ 1 ] }, { 'layer5b': truths_train[ 1 ] }, nb_epoch= 1, verbose= 1,
                    batch_size= 10, validation_data= ( features_test[ 1 ], truths_test[ 1 ] ) )

        model_c.fit( { 'main_input': features_train[ 2 ] }, { 'layer5c': truths_train[ 2 ] }, nb_epoch= 1, verbose= 1,
                    batch_size= 10, validation_data= ( features_test[ 2 ], truths_test[ 2 ] ) )

    pred_a = model_a.predict( features_test[ 0 ] )
    pred_b = model_b.predict( features_test[ 1 ] )
    pred_c = model_c.predict( features_test[ 2 ] )

    truth_a_arr.extend( labels_test[ 0 ] )
    pred_a_arr.extend( numpy.argmax( pred_a, axis= 1 ) )

    truth_b_arr.extend( labels_test[ 1 ] )
    pred_b_arr.extend( numpy.argmax( pred_b, axis= 1 ) )

    truth_c_arr.extend( labels_test[ 2 ] )
    pred_c_arr.extend( numpy.argmax( pred_c, axis= 1 ) )


from sklearn.metrics import f1_score

print 'Task 1: Primary site - Macro F1 score', f1_score( truth_a_arr, pred_a_arr, average= 'macro' )
print 'Task 1: Primary site - Micro F1 score', f1_score( truth_a_arr, pred_a_arr, average= 'micro' ) 

print 'Task 2: Tumor laterality - Macro F1 score', f1_score( truth_b_arr, pred_b_arr, average= 'macro' ) 
print 'Task 3: Tumor laterality - Micro F1 score', f1_score( truth_b_arr, pred_b_arr, average= 'micro' ) 

print 'Task 3: Histological grade - Macro F1 score', f1_score( truth_c_arr, pred_c_arr, average= 'macro' ) 
print 'Task 3: Histological grade - Micro F1 score', f1_score( truth_c_arr, pred_c_arr, average= 'micro' ) 
