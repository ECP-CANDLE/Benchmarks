import argparse
import logging

import keras
import numpy as np
import pandas as pd
from keras import backend as K
from keras import optimizers
from keras.layers import Input, Dense, Dropout, Reshape, Flatten, LocallyConnected1D, multiply
from keras.models import Model
from keras.utils import multi_gpu_model
from sklearn import preprocessing, utils, ensemble, feature_selection, model_selection

from RNAseqParse import DataLoader


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
    parser.add_argument('--epochs', type=int, default=10, help="number of epochs to do")
    parser.add_argument('--batch_size', type=int, default=1, help="batch_size")
    parser.add_argument('--lr', type=float, default=0.002, help="optmizer lr")
    parser.add_argument('--y_scale', choices=['max1', 'scale'], default='max1')
    parser.add_argument('--loss', type=str, default='mse')
    parser.add_argument('--nfeats', type=int, default=-1)
    ###############
    # model setup #
    ###############

    return parser.parse_args()


def get_activations(model, inputs, print_shape_only=False, layer_name=None):
    # Documentation is available online on Github at the address below.
    # From: https://github.com/philipperemy/keras-visualize-activations
    print('----- activations -----')
    activations = []
    inp = model.input
    if layer_name is None:
        outputs = [layer.output for layer in model.layers]
    else:
        outputs = [layer.output for layer in model.layers if layer.name == layer_name]  # all layer outputs
    print outputs
    funcs = [K.function([inp] + [K.learning_phase()], [out]) for out in outputs]  # evaluation functions
    layer_outputs = [func([inputs, 1.])[0] for func in funcs]
    for layer_activations in layer_outputs:
        activations.append(layer_activations)
        if print_shape_only:
            print(layer_activations.shape)
        else:
            print(layer_activations)
    return activations


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
    attention_mul = multiply([x_input, attention_probs], name='attention_mul')

    x = Dense(100)(attention_mul)
    x = Dense(100, activation='relu')(x)
    predictions = Dense(output_shape, activation='sigmoid')(x)
    model = Model(inputs=[x_input], outputs=predictions)
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


    print y.describe()
    y = np.array(y, dtype=np.float32)
    if args.y_scale == 'max1':
        y = np.minimum(y, np.ones(y.shape))
    elif args.y_scale == 'scale':
        print "roubust scaling"
        scaler = preprocessing.MinMaxScaler()
        shape = y.shape
        y = scaler.fit_transform(y.reshape(-1, 1)).reshape(shape)
    x = preprocessing.scale(x)
    print "Procressed y:"
    print pd.Series(y).describe()

    if args.nfeats > 0:
        rf = ensemble.RandomForestClassifier(n_estimators=2000, criterion='entropy',
                                             n_jobs=8) if args.y_scale == 'max1' or args.y_scale == 'None' else ensemble.RandomForestRegressor(
            n_estimators=2000, criterion='entropy', n_jobs=8)
        rfecv = feature_selection.RFECV(estimator=rf, step=20, cv=model_selection.StratifiedKFold(2),
                                        scoring='accuracy', n_jobs=8, min_features_to_select=args.nfeats, verbose=100)
        rfecv = rfecv.fit(x, y)
        x = rfecv.transform(x)

    labels, counts = np.unique(y, return_counts=True)
    label_dict = dict(zip(labels, counts))
    weights = create_class_weight(label_dict, y)
    print label_dict
    print weights

    model = build_model(rnaseq.shape[1], 1)
    model = multi_gpu_model(model, gpus=args.num_gpus)
    model.compile(optimizer=optimizers.Nadam(lr=args.lr),
                  loss=args.loss,
                  metrics=['accuracy', r2, 'mae', 'mse'])

    model.fit([x], y, batch_size=args.batch_size, epochs=args.epochs, validation_split=0.2, shuffle=True,
              class_weight=weights)
    print model.summary()

    attention_vector = get_activations(model, [x],
                                       print_shape_only=True,
                                       layer_name='attention_vec')
    print('attention =', attention_vector)

    pd.DataFrame(attention_vector, columns=['attention (%)']).plot(kind='bar',
                                                                   title='Attention Mechanism as '
                                                                         'a function of input'
                                                                         ' dimensions.')


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    main(arg_setup())
