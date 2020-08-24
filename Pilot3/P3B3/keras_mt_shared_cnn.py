"""
Code to export keras architecture/placeholder weights for MT CNN
Written by Mohammed Alawad
Date: 10_20_2017
"""
#np.random.seed(1337)
from keras.models import Model
from keras.layers import Dense, Dropout, Input, Embedding
from keras.layers import GlobalMaxPooling1D, Convolution1D
#from keras.layers.convolutional import Conv1D
from keras.layers.merge import Concatenate
from keras.regularizers import l2


def init_export_network(task_names,
                        task_list,
                        num_classes,
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
    model_input = Input(shape=input_shape, name="Input")
    # embedding lookup
    emb_lookup = Embedding(vocab_size,
                           wv_space,
                           input_length=in_seq_len,
                           name="embedding",
                           #embeddings_initializer=RandomUniform,
                           embeddings_regularizer=l2(emb_l2))(model_input)
    # convolutional layer and dropout
    conv_blocks = []
    for ith_filter, sz in enumerate(filter_sizes):
        conv = Convolution1D(filters=num_filters[ith_filter],
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
    for i in range(len(task_names)):
        if i in task_list:
            outlayer = Dense(num_classes[i], name=task_names[i], activation='softmax')(concat_drop)#, kernel_regularizer=l2(0.01))(concat_drop)
            FC_models.append(outlayer)
    '''
    for i in range(len(num_classes)):
        outlayer = Dense(num_classes[i], name="Dense"+str(i), activation='softmax')(concat_drop)#, kernel_regularizer=l2(0.01))(concat_drop)
        FC_models.append(outlayer)
    '''

    # the multitsk model
    model = Model(inputs=model_input, outputs=FC_models)
    model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["acc"])

    return model


if __name__ == '__main__':
    main()
