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
import talos as ta

from RNAseqParse import DataLoader
from metrics import r2


def rna_snp_pt_gridsearch(x_train, y_train, x_val, y_val, params):
    x_input = Input(shape=(x_train.shape[0],))
    x = Dense(params['first_neuron'], activation=params['activation'], kernel_initializer=params['kernel_initializer'])(
        x_input)
    layer = Dropout(params['dropout'])(x)
    encoded = Dense(params['encoded_dim'], activation=params['activation'],
                    kernel_initializer=params['kernel_initializer'])(x)

    h = None
    if params['use_attention']:
        attention_probs = Dense(params['encoded_dim'], activation='softmax', name='attention_vec')(encoded)
        attention_mul = multiply([encoded, attention_probs], name='attention_mul')
        h = x
    else:
        h = encoded
    x = Dense(params['encoded_dim'], activation=params['activation'], kernel_initializer=params['kernel_initializer'])(
        h)
    x = Dense(params['hidden_unit'])(x)
    snp_guess = Dense(y_train.shape[1], activation=params['snp_activation'])(x)

    x = Dense(params['first_neuron'], activation=params['activation'], kernel_initializer=params['kernel_initializer'])(
        encoded)
    decoded = Dense(x_train.shape[1], activation=params['last_activation'],
                    kernel_initializer=params['kernel_initializer'])(x)

    model_snps = Model(inputs=x_input, outputs=snp_guess)
    model_snps = multi_gpu_model(model_snps, 2)
    model_auto = Model(inputs=x_input, outputs=decoded)
    model_auto = multi_gpu_model(model_auto, 2)
    model_auto.compile(loss=params['auto_losses'], optimizer=optimizers.adam(), metrics=['acc', r2])
    model_snps.compile(loss=params['losses'],
                       optimizer=params['optimizer'], metrics=['accuracy', r2, 'mae', 'mse'])

    model_auto.fit(x_train, x_train, epochs=20, batch_size=200, verbose=0)

    history = model_snps.fit(x_train, y_train,
                             validation_data=[x_val, y_val],
                             batch_size=params['batch_size'],
                             epochs=params['epochs'],
                             verbose=0)

    return history, model_snps


def snps_from_rnaseq_grid_search(args):
    loader = DataLoader(args.data_path, args)
    snps, rnaseq = loader.load_aligned_snps_rnaseq(use_reduced=True, align_by=args.reduce_snps)
    rnaseq = rnaseq.set_index("Sample")
    cols = rnaseq.columns.to_series()
    index = rnaseq.index.to_series()
    rnaseq = pd.DataFrame(preprocessing.scale(rnaseq), columns=cols, index=index)
    # intersect = set(snps.columns.to_series()).intersection(set((loader.load_oncogenes_()['oncogenes'])))
    # filter_snps_oncogenes = snps[list(intersect)]
    x_big = rnaseq

    samples = set(rnaseq.index.to_series()).intersection(set(snps.index.to_series()))
    y = snps.loc[samples]
    x = rnaseq.loc[samples]
    y = y.sort_index(axis=0)
    x = x.sort_index(axis=0)

    y = y[['ENSG00000181143', 'ENSG00000145113', 'ENSG00000127914', 'ENSG00000149311']]

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
        y = scaler.fit_transform(y)
    print "Procressed y:"

    # then we can go ahead and set the parameter space
    p = {'first_neuron': (10, 2000, 10),
         'batch_size': (1, 200, 10),
         'epochs': (10, 200, 5),
         'dropout': (0, 0.3, 2),
         'kernel_initializer': ['uniform', 'normal'],
         'use_attention': [True, False],
         'encoded_dim': (10, 1000, 10),
         'auto_losses': ['mse', 'kullback_leibler_divergence', 'mae'],
         'optimizer': ['nadam', 'adam', 'SGD'],
         'losses': ['categorical_crossentropy', 'categorical_hinge', 'sparse_categorical_crossentropy'],
         'activation': ['sigmoid', 'elu', 'elu'],
         'last_activation': ['sigmoid', 'elu']}

    x = np.array(x, dtype=np.float32)
    y = np.array(x, dtype=np.float32)
    t = ta.Scan(x, y,
                params=p,
                model=rna_snp_pt_gridsearch,
                grid_downsample=0.05,
                random_method='ambient_sound',
                reduction_metric='val_r2',
                dataset_name="RNA Autoencoder pretained snp 'ENSG00000181143', 'ENSG00000145113', 'ENSG00000127914', 'ENSG00000149311'",
                experiment_no='1', val_split=0.2, debug=True, print_params=True)
    r = ta.Reporting("breast_cancer_1.csv")