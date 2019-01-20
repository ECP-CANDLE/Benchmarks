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


def arg_setup():
    parser = argparse.ArgumentParser()

    ##data
    parser.add_argument('--data_path', type=str, default="", help='Folder where data is contained.')
    parser.add_argument('--RNAseq_file', type=str, default="combined_rnaseq_data")
    parser.add_argument('--SNP_file', type=str, default='combo_snp')
    parser.add_argument('--cellline_data', type=str, default=None,
                        help='metadata file containing cell line info to add to model')
    parser.add_argument('--cache', type=str, default="cache/", help="Folder location to cache files.")
    parser.add_argument('--pooled_snps', type=str, default=None, help="Pool hdf file containing agg snps.")
    parser.add_argument('--num_gpus', type=int, default=1, help="number of gpus.")
    parser.add_argument('--epochs', type=int, default=10, help="number of epochs to do")
    parser.add_argument('--batch_size', type=int, default=1, help="batch_size")
    parser.add_argument('--lr', type=float, default=0.002, help="optmizer lr")
    parser.add_argument('--y_scale', choices=['max1', 'scale'], default='max1')
    parser.add_argument('--loss', type=str, default='mse')
    parser.add_argument('--nfeats', type=int, default=-1)
    parser.add_argument('--nfeat_step', type=int, default=100)
    parser.add_argument('--model_type',
                        choices=['rna_to_rna', 'rna_to_snp', 'rna_to_snp_pt', 'snp_to_snp', 'snp_to_rna',
                                 'grid_search'])
    parser.add_argument('--reduce_snps', type=str, default="name")
    parser.add_argument('--encoded_dim', type=int, default=100)
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
                              dropout_rate=0, regularize_genes=None, use_file_rnaseq=None):
    x_input = Input(shape=input_shape)
    print ("Input shape: ")
    print (input_shape)
    h = x_input
    to_reg = (regularize_genes is not None)

    if use_file_rnaseq is not None:
        x = h
        h = Reshape((max(input_shape), 1), input_shape=input_shape)(h)
        h = LocallyConnected1D(256, 30, strides=3, activation='relu')(h)
        h = LocallyConnected1D(256, 30, strides=3, activation='relu')(h)
        h = LocallyConnected1D(256, 30, strides=3, activation='relu')(h)
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
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return (1 - SS_res / (SS_tot + K.epsilon()))


def build_model(input_dim, output_shape):
    x_input = Input(shape=(input_dim,))
    attention_probs = Dense(input_dim, activation='softmax', name='attention_vec')(x_input)
    attention_mul = multiply([x_input, attention_probs], name='attention_mul')

    x = Dense(100)(attention_mul)
    x = Dense(100, activation='relu')(x)
    predictions = Dense(output_shape, activation='sigmoid')(x)
    model = Model(inputs=[x_input], outputs=predictions)
    return model


def build_autoencoder(input_dim, encoded_dim=1000, output_dim=1):
    x_input = Input(shape=(input_dim,))
    x = Dense(2000, activation='relu')(x_input)
    encoded = Dense(encoded_dim, activation='relu')(x)

    attention_probs = Dense(encoded_dim, activation='softmax', name='attention_vec')(encoded)
    attention_mul = multiply([encoded, attention_probs], name='attention_mul')
    x = Dense(encoded_dim, activation='relu')(attention_mul)
    x = Dense(encoded_dim)(x)
    snp_guess = Dense(output_dim, activation='sigmoid')(x)

    x = Dense(2000, activation='relu')(encoded)
    decoded = Dense(input_dim, activation='sigmoid')(x)

    model_autoencoder = Model(inputs=x_input, outputs=decoded)
    model_snp = Model(inputs=x_input, outputs=snp_guess)
    print model_autoencoder.summary()
    print model_snp.summary()
    return model_autoencoder, model_snp


def breast_cancer_model(x_train, y_train, x_val, y_val, params):
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
    model_auto.compile(loss=params['auto_losses'], optimizer=optimizers.adam, metrics=['acc'])
    model_snps.compile(loss=params['losses'],
                       optimizer=params['optimizer'], metrics=['accuracy', r2, 'mae', 'mse'])

    model_auto.fit(x_train, x_train, epochs=20, batch_size=200, verbose=0)

    history = model_snps.fit(x_train, y_train,
                             validation_data=[x_val, y_val],
                             batch_size=params['batch_size'],
                             epochs=params['epochs'],
                             verbose=0)

    return history, model_snps


def create_class_weight(labels_dict, y):
    classes = labels_dict.keys()
    weights = utils.class_weight.compute_class_weight('balanced', classes, y)
    return dict(zip(classes, weights))


def main_rna_to_snp(args):
    loader = DataLoader(args.data_path, args)
    snps, rnaseq = loader.load_aligned_snps_rnaseq(use_reduced=True, align_by=args.reduce_snps)
    rnaseq = rnaseq.set_index("Sample")

    # intersect = set(snps.columns.to_series()).intersection(set((loader.load_oncogenes_()['oncogenes'])))
    # filter_snps_oncogenes = snps[list(intersect)]

    samples = set(rnaseq.index.to_series()).intersection(set(snps.index.to_series()))
    y = snps.loc[samples]
    x = rnaseq.loc[samples]
    y = y.sort_index(axis=0)
    x = x.sort_index(axis=0)

    y = y['ENSG00000145113']

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
        rf = ensemble.RandomForestClassifier(n_estimators=1000, criterion='entropy',
                                             n_jobs=8) if args.y_scale == 'max1' or args.y_scale == 'None' else ensemble.RandomForestRegressor(
            n_estimators=1000, criterion='entropy', n_jobs=8)
        rfecv = feature_selection.RFE(estimator=rf, step=args.nfeat_step, n_features_to_select=args.nfeats, verbose=100)
        rfecv = rfecv.fit(x, y)
        x = rfecv.transform(x)

    labels, counts = np.unique(y, return_counts=True)
    label_dict = dict(zip(labels, counts))
    weights = create_class_weight(label_dict, y)
    print label_dict
    print weights

    model = build_model(x.shape[1], 1)
    if args.num_gpus >= 2:
        model = multi_gpu_model(model, gpus=args.num_gpus)
    model.compile(optimizer=optimizers.Nadam(lr=args.lr),
                  loss=args.loss,
                  metrics=['accuracy', r2, 'mae', 'mse'])
    print model.summary()

    model.fit([x], y, batch_size=args.batch_size, epochs=args.epochs, validation_split=0.2, shuffle=True,
              class_weight=weights)

    attention_vector = get_activations(model, [x],
                                       print_shape_only=True,
                                       layer_name='attention_vec')
    print('attention =', attention_vector)

    pd.DataFrame(attention_vector, columns=['attention (%)']).plot(kind='bar',
                                                                   title='Attention Mechanism as '
                                                                         'a function of input'
                                                                         ' dimensions.')


def main_rnasseq_pretrain(args):
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

    if args.nfeats > 0:
        rf = ensemble.RandomForestClassifier(n_estimators=1000, criterion='entropy',
                                             n_jobs=8) if args.y_scale == 'max1' or args.y_scale == 'None' else ensemble.RandomForestRegressor(
            n_estimators=1000, criterion='entropy', n_jobs=8)
        rfecv = feature_selection.RFE(estimator=rf, step=args.nfeat_step, n_features_to_select=args.nfeats, verbose=100)
        rfecv = rfecv.fit(x, y)
        x = rfecv.transform(x)

    #  labels, counts = np.unique(y, return_counts=True)
    # label_dict = dict(zip(labels, counts))
    # weights = create_class_weight(label_dict, y)
    # print label_dict
    # print weights

    model_auto, model_snp = build_autoencoder(x.shape[1], encoded_dim=args.encoded_dim, output_dim=4)
    if args.num_gpus >= 2:
        model_auto = multi_gpu_model(model_auto, gpus=args.num_gpus)
        model_snp = multi_gpu_model(model_snp, gpus=args.num_gpus)

    model_auto.compile(optimizer=optimizers.adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.01),
                       loss='mse',
                       metrics=['accuracy', r2, 'mae', 'mse'])
    model_snp.compile(optimizer=optimizers.adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.01),
                      loss=args.loss,
                      metrics=['accuracy', r2, 'mae', 'mse'])
    model_auto.fit(x_big, x_big, batch_size=args.batch_size, epochs=args.epochs, validation_split=0.01, shuffle=True)
    print x.shape, y.shape
    model_snp.fit(x, y, batch_size=args.batch_size, epochs=args.epochs, validation_split=0.1, shuffle=True)


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
    # p = {'first_neuron': (50, 2000, 200),
    #      'batch_size': (1, 200, 20),
    #      'epochs': (10, 200, 20),
    #      'dropout': (0, 0.3, 0.1),
    #      'kernel_initializer': ['uniform', 'normal'],
    #      'use_attention': [True, False],
    #      'encoded_dim': (10, 1000, 100),
    #      'auto_losses': [keras.losses.mse, keras.losses.kullback_leibler_divergence, keras.losses.mae],
    #      'optimizer': [keras.optimizers.nadam, keras.optimizers.adam, keras.optimizers.SGD],
    #      'losses': [keras.losses.categorical_crossentropy, keras.losses.categorical_hinge,
    #                 keras.losses.sparse_categorical_crossentropy],
    #      'activation': [keras.activations.relu, keras.activations.elu, keras.activations.sigmoid],
    #      'last_activation': [keras.activations.sigmoid, keras.activations.relu]}
    p = {'first_neuron': [200],
         'hidden_unit': [200],
         'batch_size': [200],
         'epochs': [5],
         'dropout': [0],
         'kernel_initializer': ['uniform', 'normal'],
         'use_attention': [True, False],
         'encoded_dim': [100],
         'auto_losses': ['mse', 'mae'],
         'optimizer': ['sgd'],
         'losses': ['keras.losses.categorical_crossentropy'],
         'activation': ['relu', 'elu', 'sigmoid'],
         'last_activation': ['sigmoid'],
         'snp_activation': ['sigmoid']}
    x = np.array(x, dtype=np.float32)
    y = np.array(x, dtype=np.float32)
    t = ta.Scan(x, y,
                params=p,
                model=breast_cancer_model,
                grid_downsample=1,
                random_method='quantum',
                dataset_name="RNA Autoencoder pretained snp 'ENSG00000181143', 'ENSG00000145113', 'ENSG00000127914', 'ENSG00000149311'",
                experiment_no='1', val_split=0.2, debug=True, print_params=True)
    r = ta.Reporting("breast_cancer_1.csv")


def main_snp_autoencoder(args):
    loader = DataLoader(args.data_path, args)
    snps, rnaseq = loader.load_aligned_snps_rnaseq(use_reduced=True, align_by=args.reduce_snps)
    y = snps
    y = np.minimum(y, np.ones(y.shape))

    model, _ = build_autoencoder(y.shape[1], encoded_dim=args.encoded_dim)
    if args.num_gpus >= 2:
        model = multi_gpu_model(model, gpus=args.num_gpus)
    model.compile(optimizer=optimizers.Nadam(lr=args.lr),
                  loss=args.loss,
                  metrics=['accuracy', r2, 'mae', 'mse'])
    print model.summary()

    model.fit(y, y, batch_size=args.batch_size, epochs=args.epochs, validation_split=0.1, shuffle=True)


def main_rna_autoencoder(args):
    loader = DataLoader(args.data_path, args)
    snps, rnaseq = loader.load_aligned_snps_rnaseq(use_reduced=True, align_by=args.reduce_snps)
    x = rnaseq.set_index("Sample")
    x = preprocessing.scale(x)

    model, _ = build_autoencoder(x.shape[1], encoded_dim=args.encoded_dim)
    if args.num_gpus >= 2:
        model = multi_gpu_model(model, gpus=args.num_gpus)
    model.compile(optimizer=optimizers.Nadam(lr=args.lr),
                  loss=args.loss,
                  metrics=['accuracy', r2, 'mae', 'mse'])
    print model.summary()

    model.fit(x, x, batch_size=args.batch_size, epochs=args.epochs, validation_split=0.1, shuffle=True)


def main_snp_to_rna(args):
    loader = DataLoader(args.data_path, args)
    snps, rnaseq = loader.load_aligned_snps_rnaseq(use_reduced=True, align_by=args.reduce_snps)
    x = rnaseq.set_index("Sample")
    x = preprocessing.scale(x)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    args = arg_setup()
    if args.model_type == 'rna_to_snp':
        main_rna_to_snp(args)
    elif args.model_type == 'rna_to_rna':
        main_rna_autoencoder(args)
    elif args.model_type == 'rna_to_snp_pt':
        main_rnasseq_pretrain(args)
    elif args.model_type == 'snp_to_snp':
        main_snp_autoencoder(args)
    elif args.model_type == 'snp_to_rna':
        main_snp_to_rna(args)
    elif args.model_type == 'grid_search':
        snps_from_rnaseq_grid_search(args)
