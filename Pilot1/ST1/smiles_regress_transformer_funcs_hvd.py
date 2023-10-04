############# Module Loading ##############

import argparse
import json
import os
from functools import partial

import matplotlib
import numpy as np
import pandas as pd
import ray

matplotlib.use("Agg")

import horovod.keras as hvd  # ## importing horovod to use data parallelization in another step
import keras_tuner
import tensorflow as tf
from clr_callback import *
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras.callbacks import (
    CSVLogger,
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import sequence, text

############## Defining functions #####################
######################################################


def r2(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - SS_res / (SS_tot + K.epsilon())


# Implement a Transformer block as a layer
# embed_dim: number of tokens. This is used for the key_dim for the multi-head attention calculation
# ff_dim: number of nodes in Dense layer
# epsilon: needed for numerical stability... not sure what this means to be honest


class TransformerBlock(layers.Layer):
    # __init__: defining all class variables
    def __init__(self, embed_dim, num_heads, ff_dim, rate, activation, dropout1):
        super(TransformerBlock, self).__init__()
        self.drop_chck = dropout1
        self.att = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )  # , activation=activation)
        self.ffn = keras.Sequential(
            [
                layers.Dense(ff_dim, activation=activation),
                layers.Dense(embed_dim),
            ]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    # call: building simple transformer architecture
    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        if self.drop_chck:
            attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)

        return self.layernorm2(out1 + ffn_output)


# Implement embedding layer
# Two seperate embedding layers, one for tokens, one for token index (positions).


class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions


def prep_text(texts, tokenizer, max_sequence_length):
    # Turns text into into padded sequences.
    text_sequences = tokenizer.texts_to_sequences(texts)  # turns text into tokens
    return sequence.pad_sequences(
        text_sequences, maxlen=max_sequence_length
    )  # pad all sequences so they all have same length


def model_architecture(
    embed_dim,
    num_heads,
    ff_dim,
    DR_TB,
    DR_ff,
    activation,
    dropout1,
    lr,
    loss_fn,
    hvd_switch,
):

    vocab_size = 40000  # number of possible 'words' in SMILES data
    maxlen = 250  # length of each SMILE sequence in input
    inputs = layers.Input(shape=(maxlen,))
    embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
    x = embedding_layer(inputs)
    transformer_block = TransformerBlock(
        embed_dim, num_heads, ff_dim, DR_TB, activation, dropout1
    )
    # Use 4 transformer blocks here
    x = transformer_block(x)
    x = transformer_block(x)
    x = transformer_block(x)
    x = transformer_block(x)

    x = layers.Reshape((1, 32000), input_shape=(250, 128,))(
        x
    )  # reshaping increases parameters but improves accuracy a lot
    x = layers.Dropout(DR_ff)(x)
    x = layers.Dense(1024, activation=activation)(x)
    x = layers.Dropout(DR_ff)(x)
    x = layers.Dense(256, activation=activation)(x)
    x = layers.Dropout(DR_ff)(x)
    x = layers.Dense(64, activation=activation)(x)
    x = layers.Dropout(DR_ff)(x)
    x = layers.Dense(16, activation=activation)(x)
    x = layers.Dropout(DR_ff)(x)
    outputs = layers.Dense(1, activation=activation)(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    model.summary()

    # Train and Evaluate

    opt = Adam(learning_rate=lr)

    # HVD Wrap optimizer in hvd Distributed Optimizer delegates gradient comp to original optimizer, averages gradients, and applies averaged gradients
    if hvd_switch:
        opt = hvd.DistributedOptimizer(opt)

    model.compile(loss=loss_fn, optimizer=opt, metrics=["mae", r2])
    return model


def build_model(num_heads, DR_TB, DR_ff, activation, dropout1, lr, loss_fn, hvd_switch):
    # units = hp.Int("units", min_value=32, max_value=512, step=32)
    embed_dim = 128
    ff_dim = 128
    # call existing model-building code with the hyperparameter values.
    model = model_architecture(
        embed_dim=embed_dim,
        num_heads=num_heads,
        ff_dim=ff_dim,
        DR_TB=DR_TB,
        DR_ff=DR_ff,
        activation=activation,
        dropout1=dropout1,
        lr=lr,
        loss_fn=loss_fn,
        hvd_switch=hvd_switch,
    )
    return model


def initialize_hvd(lr, x_train, y_train):
    hvd.init()
    print("I am rank %d of %d" % (hvd.rank(), hvd.size()))

    # HVD-2: GPU pinning
    gpus = tf.config.experimental.list_physical_devices("GPU")
    # Ping GPU to each9 rank
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], "GPU")

    lr = lr * hvd.size()
    x_train = np.array_split(x_train, hvd.size())
    y_train = np.array_split(y_train, hvd.size())
    return (lr, x_train, y_train)


def implement_hvd(x_train, y_train):
    x_train = x_train[hvd.rank()]
    y_train = y_train[hvd.rank()]
    return (x_train, y_train)


def callback_setting(
    hvd_switch, checkpt_file, lr, csv_file, patience_red_lr, patience_early_stop
):

    checkpointer = ModelCheckpoint(
        filepath=checkpt_file,  # "smile_regress.autosave.model.h5",
        verbose=1,
        save_weights_only=True,
        save_best_only=True,
    )

    clr = CyclicLR(base_lr=lr, max_lr=5 * lr, step_size=2000.0)

    csv_logger = CSVLogger(csv_file)  # "smile_regress.training.log")

    # learning rate tuning at each epoch
    # is it possible to do batch size tuning at each epoch as well?
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.75,
        patience=patience_red_lr,  # 20,
        verbose=1,
        mode="auto",
        epsilon=0.0001,
        cooldown=3,
        min_lr=0.000000001,
    )

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=patience_early_stop,  # 100,
        verbose=1,
        mode="auto",
    )

    if hvd_switch:
        # HVD broadcast initial variables from rank0 to all other processes
        hvd_broadcast = hvd.callbacks.BroadcastGlobalVariablesCallback(0)

        callbacks = [hvd_broadcast, reduce_lr, clr]

        if hvd.rank() == 0:
            callbacks.append(csv_logger)
            callbacks.append(early_stop)
            callbacks.append(checkpointer)

        return callbacks

    else:
        return [reduce_lr, clr, csv_logger, early_stop, checkpointer]


def build_model_tuner(hp):
    # units = hp.Int("units", min_value=32, max_value=512, step=32)
    embed_dim = 128
    num_heads = hp.Int("num_heads", min_value=12, max_value=40, step=4)
    ff_dim = 128
    DR_TB = hp.Float("DR_TB", min_value=0.025, max_value=0.5, step=0.025)
    DR_ff = hp.Float("DR_TB", min_value=0.025, max_value=0.5, step=0.025)
    activation = hp.Choice("activation", ["relu", "elu", "gelu"])
    # activation="elu"
    dropout1 = hp.Boolean("dropout_aftermulti")
    lr = hp.Float("lr", min_value=1e-6, max_value=1e-5, step=1e-6)
    loss_fn = hp.Choice("loss_fn", ["mean_squared_error", "mean_absolute_error"])
    # call existing model-building code with the hyperparameter values.
    model = model_architecture(
        embed_dim=embed_dim,
        num_heads=num_heads,
        ff_dim=ff_dim,
        DR_TB=DR_TB,
        DR_ff=DR_ff,
        activation=activation,
        dropout1=dropout1,
        lr=lr,
        loss_fn=loss_fn,
    )
    return model


# tfm.optimization.lars_optimizer.LARS(
#    learning_rate = 0.0000025,
#    momentum = 0.9,
#    weight_decay_rate = 0.0,
#    eeta = 0.001,
#    nesterov = False,
#    classic_momentum = True,
#    exclude_from_weight_decay = None,
#    exclude_from_layer_adaptation = None,
#    name = 'LARS',
#    )
