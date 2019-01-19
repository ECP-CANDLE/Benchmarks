from RNAseqParse import DataLoader
import argparse
import logging
import numpy as np
from sklearn import preprocessing, utils
import keras
from keras import backend as K
from keras.utils import plot_model

from keras import optimizers
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Conv1D,MaxPooling1D, Reshape, Flatten, LocallyConnected1D, merge
from keras.callbacks import Callback, ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler, TensorBoard
from keras.utils import get_custom_objects
from keras.utils import multi_gpu_model
import math
import os


def arg_setup():
    parser = argparse.ArgumentParser()

    ##data
    parser.add_argument('--data_path', type=str, default="",  help='Folder where data is contained.')
    parser.add_argument('--RNAseq_file', type=str, default="combined_rnaseq_data")
    parser.add_argument('--SNP_file', type=str, default='combo_snp')
    parser.add_argument('--cellline_data', type=str, default=None, help='metadata file containing cell line info to add to model')
    parser.add_argument('--cache', type=str, default="cache/", help="Folder location to cache files.")
    parser.add_argument('--pooled_snps', type=str, default=None, help="Pool hdf file containing agg snps.")
    parser.add_argument('--num_gpus', type=int, default=1, help="number of gpus.")
    ###############s
    # model setup #
    ###############

    return parser.parse_args()


def build_feature_model_genes(input_shape, name='', dense_layers=[500, 500],
                        activation='relu', residual=False,
                        dropout_rate=0,  regularize_genes=None, use_file_rnaseq=None):
    x_input = Input(shape=input_shape)
    print ("Input shape: ")
    print (input_shape)
    h = x_input
    to_reg = (regularize_genes is not None)

    if use_file_rnaseq is not None:
        x = h
        h = Reshape((max(input_shape), 1), input_shape=input_shape)(h)
        h = LocallyConnected1D(256, 30, strides=3, activation='relu')(h)
        h =  LocallyConnected1D(256, 30, strides=3, activation='relu')(h)
        h = LocallyConnected1D(256, 30, strides =3, activation='relu')(h)
        h = Flatten()(h)
    for i, layer in enumerate(dense_layers):
        x = h
        if to_reg:
            h = Dense(layer, activation=activation, kernel_regularizer=regularize_genes)(h)
            to_reg = False
        else:
            h = Dense(layer, activation=activation)(h)
        if dropout_rate > 0:
            h = Dropout(dropout_rate)(h)
        if residual:
            try:
                h = keras.layers.add([h, x])
            except ValueError:
                pass
    model = Model(x_input, h, name=name)
    return model

def r2(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true-y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

def build_model(input_dim, output_shape):
    x_input = Input(shape= (input_dim, ))
    attention_probs = Dense(input_dim, activation='softmax', name='attention_vec')(x_input)
    attention_mul = merge([x_input, attention_probs], output_shape=32, name='attention_mul', mode='mul')

    x = Dense(64)(attention_mul)
    x = Dense(64,  activation='relu')(x_input)
    x = Dense(64,  activation='relu')(x)
    x = Dense(64,  activation='relu')(x)
    predictions = Dense(output_shape, activation='sigmoid')(x)
    model = Model(inputs=x_input, outputs=predictions)
    return model

def create_class_weight(labels_dict, y):
    classes = labels_dict.keys()
    weights = utils.class_weight.compute_class_weight('balanced', classes, y)
    return dict(zip(classes, weights))


def main(args):
    loader = DataLoader(args.data_path, args)
    snps, rnaseq = loader.load_aligned_snps_rnaseq(use_reduced=True)
    rnaseq = rnaseq.set_index("Sample")

    # intersect = set(snps.columns.to_series()).intersection(set((loader.load_oncogenes_()['oncogenes'])))
    # filter_snps_oncogenes = snps[list(intersect)]

    samples = set(rnaseq.index.to_series()).intersection(set(snps.index.to_series()))
    y = snps.loc[samples]
    x = rnaseq.loc[samples]
    y = y.sort_index(axis=0)
    x = x.sort_index(axis=0)

    y=y['ENSG00000145113']

    print y.tail()
    print x.tail()
    print x.shape, y.shape

    model = build_model(rnaseq.shape[1], 1)
    model = multi_gpu_model(model, gpus=args.num_gpus)
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy', r2, 'mae', 'mse'] )
    print model.summary()
    print y.describe()
    y = np.array(y, dtype=np.float32)
    y = np.minimum(y, np.ones(y.shape))
    x = preprocessing.scale(x)

    labels, counts = np.unique(y, return_counts=True)
    label_dict = dict(zip(labels, counts))
    weights = create_class_weight(label_dict, y)
    print label_dict
    print weights
    plot_model(model, to_file='model.png')
    model.fit(x, y, batch_size=1, epochs=50, validation_split=0.2, shuffle=True, class_weight=weights)



if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    main(arg_setup())
