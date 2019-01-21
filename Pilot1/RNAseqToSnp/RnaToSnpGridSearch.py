from __future__ import print_function

import argparse
import logging

from comet_ml import Experiment


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
from sklearn.model_selection import train_test_split
from RNAseqParse import DataLoader
from metrics import r2

# import comet_ml in the top of your file

# Add the following code anywhere in your machine learning file

###################
# RNA GRIDSEARRCH #
###################

gpu_nums = None
x_big = None


def my_product(inp):
    from itertools import product
    return (dict(zip(inp.keys(), values)) for values in product(*inp.values()))


def do_search(x_train, y_train, x_val, y_val, model, params):
    perms = my_product(params)

    for perm in perms:
        print(perms)
        history, _ = model(x_train, y_train, x_val, y_val, perm)
        print("MODEL DONE: ", str(history.history['r2'][-1]), str(history.history['val_r2'][-1]),
              str(history.history['val_acc'][-1]))


def rna_rna_gridsearch_model(x_train, y_train, x_val, y_val, params):
    x_input = Input(shape=(x_train.shape[1],))
    x = Dense(params['first_neuron'], activation=params['activation'], kernel_initializer=params['kernel_initializer'])(
        x_input)
    layer = Dropout(params['dropout'])(x)
    encoded = Dense(params['encoded_dim'], activation=params['activation'],
                    kernel_initializer=params['kernel_initializer'])(x)

    x = Dense(params['first_neuron'], activation=params['activation'], kernel_initializer=params['kernel_initializer'])(
        encoded)
    decoded = Dense(x_train.shape[1], activation=params['last_activation'],
                    kernel_initializer=params['kernel_initializer'])(x)

    model_auto = Model(inputs=x_input, outputs=decoded)
    print(gpu_nums)
    model_auto = multi_gpu_model(model_auto, gpus=gpu_nums)
    model_auto.compile(loss=params['auto_losses'], optimizer=params['optimizer'](lr=params['lr']), metrics=['acc', r2])

    history = model_auto.fit(x_train, y_train, validation_data=[x_val, y_val], epochs=params['epochs'],
                             batch_size=params['batch_size'], verbose=0)

    return history, model_auto


def rna_rna_gridsearch_params():
    params = {'first_neuron': [2000, 3000],
              'batch_size': (100, 400, 4),
              'epochs': (10, 50, 2),
              'dropout': (0, 0.3, 3),
              'kernel_initializer': ['uniform', 'normal'],
              'encoded_dim': [500, 1000, 1500, 2000],
              'auto_losses': ['mse', 'kullback_leibler_divergence', 'mae'],
              'optimizer': [keras.optimizers.adam, keras.optimizers.SGD],
              'lr': [1.0, 0.1, 0.001],
              'activation': ['sigmoid', 'relu'],
              'last_activation': ['sigmoid', 'relu']}
    return params


##based on analysis

def rna_rna_gridsearch_params_2():
    params = {'first_neuron': [2000, 3000],
              'batch_size': (100, 400, 4),
              'epochs': (10, 50, 2),
              'dropout': (0, 0.3, 3),
              'kernel_initializer': ['uniform', 'normal'],
              'encoded_dim': [500, 1000, 1500, 2000],
              'auto_losses': ['mse', 'kullback_leibler_divergence', 'mae'],
              'optimizer': [keras.optimizers.adam, keras.optimizers.SGD],
              'lr': [1.0, 0.1, 0.001],
              'activation': ['sigmoid', 'relu'],
              'last_activation': ['sigmoid', 'relu']}
    return params


def rna_rna_gridsearch(args):
    global gpu_nums
    gpu_nums = args.num_gpus
    loader = DataLoader(args.data_path, args)
    _, rnaseq = loader.load_aligned_snps_rnaseq(use_reduced=True, align_by=args.reduce_snps)
    rnaseq = rnaseq.set_index("Sample")
    x = preprocessing.scale(rnaseq)

    x = np.array(x, dtype=np.float32)
    y = np.array(x, dtype=np.float32)

    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2)

    t = ta.Scan(x_train, y_train, x_val=x_val, y_val=y_val,
                params=rna_rna_gridsearch_params(),
                model=rna_rna_gridsearch_model,
                grid_downsample=0.01,
                #       reduction_metric='val_r2',
                #       reduction_method='correlation',
                dataset_name="RNA_Autoencoder",
                experiment_no='1', debug=True, print_params=True)
    r = ta.Reporting("rna_autoencoder.csv")


##############
# SNP -> SNP #
##############

def snp_snp_gridsearch_model(x_train, y_train, x_val, y_val, params):
    experiment = Experiment(api_key="sWqygZPzck6CCDVasK2e0PHhT",
                            project_name="general", workspace="aclyde11")
    print(x_train.shape, y_train.shape, x_val.shape, y_val.shape)
    x_input = Input(shape=(x_train.shape[1],))
    x = Dense(params['first_neuron'], activation=params['activation'], kernel_initializer=params['kernel_initializer'])(
        x_input)
    layer = Dropout(params['dropout'])(x)
    encoded = Dense(params['encoded_dim'], activation=params['activation'],
                    kernel_initializer=params['kernel_initializer'])(x)

    x = Dense(params['first_neuron'], activation=params['activation'], kernel_initializer=params['kernel_initializer'])(
        encoded)
    decoded = Dense(x_train.shape[1], activation=params['last_activation'],
                    kernel_initializer=params['kernel_initializer'])(x)

    model_auto = Model(inputs=x_input, outputs=decoded)
    print(gpu_nums)
    if gpu_nums > 1:
        model_auto = multi_gpu_model(model_auto, gpus=gpu_nums)
    model_auto.compile(loss=params['auto_losses'], optimizer=params['optimizer'](lr=params['lr']), metrics=['acc', r2])

    history = model_auto.fit(x_train, y_train, validation_data=[x_val, y_val], epochs=params['epochs'],
                             batch_size=params['batch_size'], verbose=0)

    return history, model_auto


def snp_snp_gridsearch_params():
    params = {'first_neuron': [1750],
              'batch_size': [150],
              'epochs': [70],
              'dropout': [0.2],
              'kernel_initializer': ['uniform'],
              'encoded_dim': [350, 500, 750],
              'auto_losses': ['categorical_crossentropy'],
              'optimizer': [keras.optimizers.adam],
              'lr': [0.001, 0.0005],
              'activation': ['relu'],
              'last_activation': ['sigmoid']}
    return params


def snp_snp_gridsearch(args):
    global gpu_nums
    gpu_nums = args.num_gpus
    loader = DataLoader(args.data_path, args)
    y, _ = loader.load_aligned_snps_rnaseq(use_reduced=True, align_by=args.reduce_snps)

    print(y.tail())
    print(y.describe())
    y = np.array(y, dtype=np.float32)
    if args.y_scale == 'max1':
        y = np.minimum(y, np.ones(y.shape))
    elif args.y_scale == 'scale':
        print("roubust scaling")
        scaler = preprocessing.MinMaxScaler()
        shape = y.shape
        y = scaler.fit_transform(y)
    print("Procressed y:")

    x = np.array(y, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2)
    do_search(x_train, y_train, x_val, y_val, snp_snp_gridsearch_model, snp_snp_gridsearch_params())






#######################
# RNA->SNP GRIDSEARCH #
#######################

def rna_snp_pt_gridsearch(x_train, y_train, x_val, y_val, params):
    x_input = Input(shape=(x_train.shape[1],))
    x = Dense(params['first_neuron'], activation=params['activation'], kernel_initializer=params['kernel_initializer'])(
        x_input)
    layer = Dropout(params['dropout'])(x)
    encoded = Dense(params['encoded_dim'], activation=params['activation'],
                    kernel_initializer=params['kernel_initializer'])(x)

    h = None
    if params['use_attention']:
        attention_probs = Dense(params['encoded_dim'], activation='softmax', name='attention_vec')(encoded)
        attention_mul = multiply([encoded, attention_probs], name='attention_mul')
        h = attention_mul
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
    model_snps = multi_gpu_model(model_snps, gpu_nums)
    model_auto = Model(inputs=x_input, outputs=decoded)
    model_auto = multi_gpu_model(model_auto, gpu_nums)

    model_auto.compile(loss=params['auto_losses'], optimizer=params['auto_optimizer'](params['lr'] * 4),
                       metrics=['acc', r2])
    model_auto.fit(x_big, x_big, epochs=2, batch_size=200, verbose=1, validation_split=0.05)

    model_snps.compile(loss=params['snp_losses'],
                       optimizer=params['snp_optimizer']('lr'), metrics=['accuracy', r2, 'mae', 'mse'])
    print("TRAINING MODEL")



    history = model_snps.fit(x_train, y_train,
                             validation_data=[x_val, y_val],
                             batch_size=params['batch_size'],
                             epochs=params['epochs'],
                             verbose=1)
    print("returning...")
    return history, model_snps


def snps_from_rnaseq_params():
    p = {'first_neuron': [1000, 1500],
         'batch_size': [200],
         'epochs': [50],
         'dropout': (0, 0.3, 2),
         'hidden_unit': ([500, 750]),
         'kernel_initializer': ['uniform'],
         'use_attention': [True, False],
         'encoded_dim': [500, 750, 1000],
         'auto_losses': ['mse', 'mae'],
         'snp_losses': ['categorical_crossentropy', 'sparse_categorical_crossentropy'],
         'snp_optimizer': [keras.optimizers.adam],
         'auto_optimizer': [keras.optimizers.adam],
         'lr': [0.001, 0.01, 0.1],
         'activation': ['sigmoid', 'elu', 'relu'],
         'last_activation': ['sigmoid', 'relu'],
         'snp_activation': ['sigmoid']}
    return p

def snps_from_rnaseq_grid_search(args):
    global gpu_nums
    gpu_nums = args.num_gpus
    loader = DataLoader(args.data_path, args)
    snps, rnaseq = loader.load_aligned_snps_rnaseq(use_reduced=True, align_by=args.reduce_snps)
    rnaseq = rnaseq.set_index("Sample")
    cols = rnaseq.columns.to_series()
    index = rnaseq.index.to_series()
    rnaseq = pd.DataFrame(preprocessing.scale(rnaseq), columns=cols, index=index)
    # intersect = set(snps.columns.to_series()).intersection(set((loader.load_oncogenes_()['oncogenes'])))
    # filter_snps_oncogenes = snps[list(intersect)]
    global x_big
    x_big = rnaseq

    samples = set(rnaseq.index.to_series()).intersection(set(snps.index.to_series()))
    y = snps.loc[samples]
    x = rnaseq.loc[samples]
    y = y.sort_index(axis=0)
    x = x.sort_index(axis=0)

    y = y[['ENSG00000181143', 'ENSG00000145113', 'ENSG00000127914', 'ENSG00000149311']]

    print(y.tail())
    print(x.tail())
    print(x.shape, y.shape)

    print(y.describe())
    y = np.array(y, dtype=np.float32)
    if args.y_scale == 'max1':
        y = np.minimum(y, np.ones(y.shape))
    elif args.y_scale == 'scale':
        print("roubust scaling")
        scaler = preprocessing.MinMaxScaler()
        shape = y.shape
        y = scaler.fit_transform(y)
    print("Procressed y:")

    # then we can go ahead and set the parameter space


    x = np.array(x, dtype=np.float32)
    y = np.array(x, dtype=np.float32)

    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.15)

    t = ta.Scan(x_train, y_train, x_val=x_val, y_val=y_val,
                params=snps_from_rnaseq_params(),
                model=rna_snp_pt_gridsearch,
                grid_downsample=0.1,
                reduction_metric='val_r2',
                dataset_name="RNA Autoencoder pretained snp 'ENSG00000181143', 'ENSG00000145113', 'ENSG00000127914', 'ENSG00000149311'",
                experiment_no='1', debug=True, print_params=True)
