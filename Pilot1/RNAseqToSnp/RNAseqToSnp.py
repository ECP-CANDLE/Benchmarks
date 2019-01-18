from RNAseqParse import DataLoader
import argparse
import logging
import keras
from keras import backend as K
from keras import optimizers
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Conv1D,MaxPooling1D, Reshape, Flatten, LocallyConnected1D
from keras.callbacks import Callback, ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler, TensorBoard
from keras.utils import get_custom_objects
from keras.utils import multi_gpu_model

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

def build_model(input_shape_feats, output_shape):
    x_input = Input(shape=(input_shape_feats, ))
    x = Reshape((max(input_shape_feats), 1))(x_input)
    x = LocallyConnected1D(256, 10, strides=2, activation='relu')(x)
    x = LocallyConnected1D(256, 10, strides=2, activation='relu')(x)
    x = LocallyConnected1D(256, 10, strides=2, activation='relu')(x)
    x = LocallyConnected1D(256, 10, strides=2, activation='relu')(x)
    x = Flatten()(x)
    x = Dense(5000)(x)
    x = Dense(5000)(x)
    predictions = Dense(output_shape[1], activation='sigmoid')(x)
    model = Model(inputs=x_input, outputs=predictions)
    return model


def main(args):
    loader = DataLoader(args.data_path, args)
    snps, rnaseq = loader.load_aligned_snps_rnaseq(use_reduced=True)
    print snps
    print rnaseq
 #   model = build_model(rnaseq.shape[1])
 #   model = multi_gpu_model(model, gpus=2)
 #   model.compile(optimizer='rmsprop',
 #                 loss='mae',
 #                 metrics=['accuracy', r2])


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    main(arg_setup())
