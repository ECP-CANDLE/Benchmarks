"""
Code to export keras architecture/placeholder weights for MT CNN
Written by Mohammed Alawad
Date: 10_20_2017
"""
"""
DISCLAIMER

UT-BATTELLE, LLC AND THE GOVERNMENT MAKE NO REPRESENTATIONS AND DISCLAIM ALL WARRANTIES, 
BOTH EXPRESSED AND IMPLIED.  THERE ARE NO EXPRESS OR IMPLIED WARRANTIES OF MERCHANTABILITY 
OR FITNESS FOR A PARTICULAR PURPOSE, OR THAT THE USE OF THE SOFTWARE WILL NOT INFRINGE ANY 
PATENT, COPYRIGHT, TRADEMARK, OR OTHER PROPRIETARY RIGHTS, OR THAT THE SOFTWARE WILL 
ACCOMPLISH THE INTENDED RESULTS OR THAT THE SOFTWARE OR ITS USE WILL NOT RESULT IN INJURY 
OR DAMAGE.  THE USER ASSUMES RESPONSIBILITY FOR ALL LIABILITIES, PENALTIES, FINES, CLAIMS, 
CAUSES OF ACTION, AND COSTS AND EXPENSES, CAUSED BY, RESULTING FROM OR ARISING OUT OF, IN 
WHOLE OR IN PART THE USE, STORAGE OR DISPOSAL OF THE SOFTWARE.
"""
"""
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, 
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR 
PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE 
FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR 
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.
"""

import numpy as np
#np.random.seed(1337)
from keras.models import Model
from keras.layers import Dense, Activation, Dropout, Flatten, Input, Embedding
from keras.layers import merge as Merge
from keras.layers import GlobalMaxPooling1D, Convolution1D
#from keras.layers.convolutional import Conv1D
from keras.layers.merge import Concatenate
from keras import optimizers
import keras.backend as K
from keras.regularizers import l2
import pickle
import argparse
import os
from keras.initializers import RandomUniform,lecun_uniform

def init_export_network(num_classes,
                        in_seq_len,
                        vocab_size,
                        wv_space,
                        filter_sizes,
                        num_filters,
                        concat_dropout_prob,
                        emb_l2,
                        w_l2,
                        optimizer):


    # define network layers ----------------------------------------------------
    input_shape = tuple([in_seq_len])
    model_input = Input(shape=input_shape, name= "Input")
    # embedding lookup
    emb_lookup = Embedding(vocab_size,
                           wv_space,
                           input_length=in_seq_len,
                           name="embedding",
                           #embeddings_initializer=RandomUniform,
                           embeddings_regularizer=l2(emb_l2))(model_input)
    # convolutional layer and dropout
    conv_blocks = []
    for ith_filter,sz in enumerate(filter_sizes):
        conv = Convolution1D(filters=num_filters[ ith_filter ],
                             kernel_size=sz,
                             padding="same",
                             activation="relu",
                             strides=1,
                             # kernel_initializer ='lecun_uniform,
                             name=str(ith_filter) + "_thfilter")(emb_lookup)
        conv_blocks.append(GlobalMaxPooling1D()(conv))
    concat = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]
    concat_drop = Dropout(concat_dropout_prob)(concat)

    # different dense layer per tasks
    FC_models = []
    for i in range(len(num_classes)):
        outlayer = Dense(num_classes[i], name= "Dense"+str(i), activation='softmax')( concat_drop )#, kernel_regularizer=l2(0.01))( concat_drop )
        FC_models.append(outlayer)


    # the multitsk model
    model = Model(inputs=model_input, outputs = FC_models)
    model.compile( loss= "sparse_categorical_crossentropy", optimizer= optimizer, metrics=[ "acc" ] )

    return model

if __name__ == '__main__':
    main()

