#! /usr/bin/env python


import numpy as np

import keras
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Dense, Dropout
from keras.callbacks import Callback
from keras import regularizers
from keras.metrics import mean_squared_error, mean_absolute_error

import candle


def r2_heteroscedastic(y_true, y_pred):
    y_out = K.reshape(y_pred[:,:-1], K.shape(y_true))
    SS_res =  K.sum(K.square(y_true - y_out))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return (1 - SS_res/(SS_tot + K.epsilon()))


def mae_heteroscedastic(y_true, y_pred):
    y_out = K.reshape(y_pred[:,:-1], K.shape(y_true))
    return mean_absolute_error(y_true, y_out)

def mse_heteroscedastic(y_true, y_pred):
    y_out = K.reshape(y_pred[:,:-1], K.shape(y_true))
    return mean_squared_error(y_true, y_out)

def meanS_heteroscesdastic(y_true, y_pred):
    log_sig2 = y_pred[:,1]
    return K.mean(log_sig2)

def quantile_loss(quantile, y_true, y_pred):
    error = (y_true - y_pred)
    return K.mean(K.maximum(quantile*error, (quantile-1)*error), axis=-1)

def quantile50(y_true, y_pred):
    y_out0 = K.reshape(y_pred[:,0], K.shape(y_true))
    error = (y_true-y_out0)
    quantile = 0.5
    return quantile_loss(quantile, y_true, y_out0)


def quantile10(y_true, y_pred):
    y_out1 = K.reshape(y_pred[:,1], K.shape(y_true))
    error = (y_true-y_out1)
    quantile = 0.1
    return quantile_loss(quantile, y_true, y_out1)


def quantile90(y_true, y_pred):
    y_out2 = K.reshape(y_pred[:,2], K.shape(y_true))
    error = (y_true-y_out2)
    quantile = 0.9
    return quantile_loss(quantile, y_true, y_out2)


class ModelRecorder(Callback):
    def __init__(self, save_all_models=False):
        Callback.__init__(self)
        self.save_all_models = save_all_models
        candle.register_permanent_dropout()

    def on_train_begin(self, logs={}):
        self.val_losses = []
        self.best_val_loss = np.Inf
        self.best_model = None

    def on_epoch_end(self, epoch, logs={}):
        val_loss = logs.get('val_loss')
        self.val_losses.append(val_loss)
        if val_loss < self.best_val_loss:
            self.best_model = keras.models.clone_model(self.model)
            self.best_val_loss = val_loss


class SimpleWeightSaver(Callback):    

    def __init__(self, fname):
        self.fname = fname

    def set_model(self, model):
        if isinstance(model.layers[-2], Model):
            self.model = model.layers[-2]
        else:
            self.model = model

    def on_train_end(self, logs={}):
        self.model.save_weights(self.fname)


def build_model(loader, args, logger=None, permanent_dropout=True, silent=False):
    if args.loss == 'heteroscedastic':
        model = build_heteroscedastic_model(loader, args, logger, permanent_dropout, silent)
    elif args.loss == 'quantile':
        model = build_quantile_model(loader, args, logger, permanent_dropout, silent)
    else:
        model = build_homoscedastic_model(loader, args, logger, permanent_dropout, silent)

    return model

def build_feature_model(input_shape, name='', dense_layers=[1000, 1000],
                        activation='relu', residual=False,
                        dropout_rate=0, permanent_dropout=True,
                        reg_l2=0):
    x_input = Input(shape=input_shape)
    h = x_input
    for i, layer in enumerate(dense_layers):
        x = h
        if reg_l2 > 0:
            h = Dense(layer, activation=activation, kernel_regularizer=regularizers.l2(reg_l2))(h)
        else:
            h = Dense(layer, activation=activation)(h)
        if dropout_rate > 0:
            if permanent_dropout:
                h = candle.PermanentDropout(dropout_rate)(h)
            else:
                h = Dropout(dropout_rate)(h)
        if residual:
            try:
                h = keras.layers.add([h, x])
            except ValueError:
                pass
    model = Model(x_input, h, name=name)
    return model


def build_homoscedastic_model(loader, args, logger=None, permanent_dropout=True, silent=False):
    input_models = {}
    dropout_rate = args.drop
    reg_l2 =  args.reg_l2
    for fea_type, shape in loader.feature_shapes.items():
        base_type = fea_type.split('.')[0]
        if base_type in ['cell', 'drug']:
            box = build_feature_model(input_shape=shape, name=fea_type,
                                      dense_layers=args.dense_feature_layers,
                                      dropout_rate=dropout_rate, permanent_dropout=permanent_dropout,
                                      reg_l2=reg_l2)
            if not silent:
                logger.debug('Feature encoding submodel for %s:', fea_type)
                box.summary(print_fn=logger.debug)
            input_models[fea_type] = box

    inputs = []
    encoded_inputs = []
    for fea_name, fea_type in loader.input_features.items():
        shape = loader.feature_shapes[fea_type]
        fea_input = Input(shape, name='input.'+fea_name)
        inputs.append(fea_input)
        if fea_type in input_models:
            input_model = input_models[fea_type]
            encoded = input_model(fea_input)
        else:
            encoded = fea_input
        encoded_inputs.append(encoded)

    merged = keras.layers.concatenate(encoded_inputs)

    h = merged
    for i, layer in enumerate(args.dense):
        x = h
        if reg_l2 > 0:
            h = Dense(layer, activation=args.activation, kernel_regularizer=regularizers.l2(reg_l2))(h)
        else:
            h = Dense(layer, activation=args.activation)(h)
        if dropout_rate > 0:
            if permanent_dropout:
                h = candle.PermanentDropout(dropout_rate)(h)
            else:
                h = Dropout(dropout_rate)(h)
        if args.residual:
            try:
                h = keras.layers.add([h, x])
            except ValueError:
                pass
    output = Dense(1)(h)

    return Model(inputs, output)


def build_heteroscedastic_model(loader, args, logger=None, permanent_dropout=True, silent=False):
    input_models = {}
    dropout_rate = args.drop
    reg_l2 =  args.reg_l2
    for fea_type, shape in loader.feature_shapes.items():
        base_type = fea_type.split('.')[0]
        if base_type in ['cell', 'drug']:
            box = build_feature_model(input_shape=shape, name=fea_type,
                                        dense_layers=args.dense_feature_layers,
                                        dropout_rate=dropout_rate, permanent_dropout=permanent_dropout,
                                        reg_l2=reg_l2)
            if not silent:
                logger.debug('Feature encoding submodel for %s:', fea_type)
                box.summary(print_fn=logger.debug)
            input_models[fea_type] = box

    inputs = []
    encoded_inputs = []
    for fea_name, fea_type in loader.input_features.items():
        shape = loader.feature_shapes[fea_type]
        fea_input = Input(shape, name='input.'+fea_name)
        inputs.append(fea_input)
        if fea_type in input_models:
            input_model = input_models[fea_type]
            encoded = input_model(fea_input)
        else:
            encoded = fea_input
        encoded_inputs.append(encoded)

    merged = keras.layers.concatenate(encoded_inputs)

    h = merged
    for i, layer in enumerate(args.dense):
        x = h
        if reg_l2 > 0:
            h = Dense(layer, activation=args.activation, kernel_regularizer=regularizers.l2(reg_l2))(h)
        else:
            h = Dense(layer, activation=args.activation)(h)
        if dropout_rate > 0:
            if permanent_dropout:
                h = candle.PermanentDropout(dropout_rate)(h)
            else:
                h = Dropout(dropout_rate)(h)
        if args.residual:
            try:
                h = keras.layers.add([h, x])
            except ValueError:
                pass
    output = Dense(2, bias_initializer='ones')(h)

    return Model(inputs, output)

def build_quantile_model(loader, args, logger=None, permanent_dropout=True, silent=False):
    input_models = {}
    dropout_rate = args.drop
    reg_l2 =  args.reg_l2
    for fea_type, shape in loader.feature_shapes.items():
        base_type = fea_type.split('.')[0]
        if base_type in ['cell', 'drug']:
            box = build_feature_model(input_shape=shape, name=fea_type,
                                        dense_layers=args.dense_feature_layers,
                                        dropout_rate=dropout_rate,
                                        permanent_dropout=permanent_dropout,
                                        reg_l2=reg_l2)
            if not silent:
                logger.debug('Feature encoding submodel for %s:', fea_type)
                box.summary(print_fn=logger.debug)
            input_models[fea_type] = box

    inputs = []
    encoded_inputs = []
    for fea_name, fea_type in loader.input_features.items():
        shape = loader.feature_shapes[fea_type]
        fea_input = Input(shape, name='input.'+fea_name)
        inputs.append(fea_input)
        if fea_type in input_models:
            input_model = input_models[fea_type]
            encoded = input_model(fea_input)
        else:
            encoded = fea_input
        encoded_inputs.append(encoded)

    merged = keras.layers.concatenate(encoded_inputs)

    h = merged
    for i, layer in enumerate(args.dense):
        x = h
        h = Dense(layer, activation=args.activation, kernel_regularizer=regularizers.l2(args.reg_l2))(h)
        if dropout_rate > 0:
            if permanent_dropout:
                h = candle.PermanentDropout(dropout_rate)(h)
            else:
                h = Dropout(dropout_rate)(h)
        if args.residual:
            try:
                h = keras.layers.add([h, x])
            except ValueError:
                pass
    output = Dense(3, bias_initializer='ones')(h)

    return Model(inputs, output)


def heteroscedastic_loss(y_true, y_pred):
    y_shape = K.shape(y_true)
    y_out = K.reshape(y_pred[:,0], y_shape)
    diff_sq = K.square(y_out - y_true)
    log_sig2 = y_pred[:,1]
    
    return K.mean(K.exp(-log_sig2) * diff_sq + log_sig2)


def tilted_loss(quantile, y_true, f):
    error = (y_true-f)
    return K.mean(K.maximum(quantile*error, (quantile-1)*error), axis=-1)


def triple_quantile_loss(y_true, y_pred):
    y_shape = K.shape(y_true)
    y_out0 = K.reshape(y_pred[:,0], y_shape)
    y_out1 = K.reshape(y_pred[:,1], y_shape)
    y_out2 = K.reshape(y_pred[:,2], y_shape)

    return tilted_loss(0.1, y_true, y_out1) + tilted_loss(0.9, y_true, y_out2) + 2. * tilted_loss(0.5, y_true, y_out0)
