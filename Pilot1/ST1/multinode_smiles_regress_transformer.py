#!/usr/bin/env python

import pandas as pd
import numpy as np
import os
import sys
import gzip
import argparse
import socket
import json
import math

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing import text

from pbsutils import tf_config, pmi_rank, pmi_size

# set up some logging
from datetime import datetime as dt
def now():
    now = dt.now()
    return '{}'.format(dt.now())

thisdir = os.path.dirname(os.path.realpath(__file__))
logfile_name = '{}/{}.log'.format(thisdir, pmi_rank())
f = open(logfile_name, 'w')

# needed on polaris
if 'http_proxy' in os.environ:
    del os.environ['http_proxy']
if 'https_proxy' in os.environ:
    del os.environ['https_proxy']

## set up tensorflow
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# RBDGX tf_config
if socket.gethostname().startswith('rbdgx1'):
    index = 0
    tf_config = {
            'cluster': {
                'worker': ['192.168.200.101:12345', '192.168.200.103:12346']
                },
            'task': {'type': 'worker', 'index': index}
            }

elif socket.gethostname().startswith('rbdgx2'):
    index = 1
    tf_config = {
            'cluster': {
                'worker': ['192.168.200.101:12345', '192.168.200.103:12346']
                },
            'task': {'type': 'worker', 'index': index}
            }

# Polaris tf_config
else:
    tf_config = tf_config()


os.environ['TF_CONFIG'] = json.dumps(tf_config)
num_workers = len(tf_config['cluster']['worker'])
f.write ('{}: num workers {}\n'.format(now(), num_workers))
f.write('{}: TF_CONFIG: {}\n'.format(now(), os.environ['TF_CONFIG']))

communication_options = tf.distribute.experimental.CommunicationOptions(
    #bytes_per_pack=50 * 1024 * 1024,
    #timeout_seconds=120.0,
    implementation=tf.distribute.experimental.CommunicationImplementation.NCCL
)
strategy = tf.distribute.MultiWorkerMirroredStrategy(
        communication_options=communication_options,
    )
f.write('{}: tensorflow version: {}\n'.format(now(), tf.__version__))
f.write('{}: Number of devices: {}\n'.format(now(), strategy.num_replicas_in_sync))

## get args / globals

psr = argparse.ArgumentParser(description='input csv file')
psr.add_argument('--in_train',  default='in_train')
psr.add_argument('--in_vali',  default='in_vali')
psr.add_argument('--ep',  type=int, default=400)
args=vars(psr.parse_args())
f.write('args: {}'.format(args))

EPOCH = args['ep']
BATCH = 32
GLOBAL_BATCH_SIZE = BATCH * strategy.num_replicas_in_sync

f.write("{}: batch size: {}\n".format(now(), BATCH))
f.write("{}: global batch size {}\n".format(now(), GLOBAL_BATCH_SIZE))

data_path_train = args['in_train']
data_path_vali = args['in_vali']

DR    = 0.1      # Dropout rate                  

## define r2 for reporting
def r2(y_true, y_pred):
    SS_res =  K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return (1 - SS_res/(SS_tot + K.epsilon()))

## Implement a Transformer block as a layer
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

## Implement embedding layer
## Two seperate embedding layers, one for tokens, one for token index (positions).
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


## Input and prepare dataset
vocab_size = 40000  # 
maxlen = 250  # 

f.write('{}: reading training data {}\n'.format(now(), data_path_train))
data_train = pd.read_csv(data_path_train)
f.write('{}: reading validation data {}\n'.format(now(), data_path_vali))
data_vali = pd.read_csv(data_path_vali)

f.write('{}\n'.format(data_train.head()))
f.write('{}\n'.format(data_vali.head()))

y_train = data_train["type"].values.reshape(-1, 1) * 1.0
y_val = data_vali["type"].values.reshape(-1, 1) * 1.0

f.write('{}\n'.format(y_train))
f.write('{}\n'.format(y_val))

f.write('{}: calling tokenizer.fit with num_words: {}\n'.format(now(), vocab_size))
tokenizer = text.Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(data_train["smiles"])

def prep_text(texts, tokenizer, max_sequence_length):
    # Turns text into into padded sequences.
    text_sequences = tokenizer.texts_to_sequences(texts)
    return sequence.pad_sequences(text_sequences, maxlen=maxlen)

x_train = prep_text(data_train["smiles"], tokenizer, maxlen)
x_val = prep_text(data_vali["smiles"], tokenizer, maxlen)

f.write('{}: x_train.shape {}\n'.format(now(), x_train.shape))
f.write('{}: y_train.shape {}\n'.format(now(), y_train.shape))

steps = x_train.shape[0]//GLOBAL_BATCH_SIZE
validation_steps = x_val.shape[0]//GLOBAL_BATCH_SIZE

f.write('{}: samples {}, global_batch_size {}, steps {}\n'.format(now(), x_train.shape[0], GLOBAL_BATCH_SIZE, steps))
f.write('{}: val samples {}, global_batch_size {}, val_steps {}\n'.format(now(), x_val.shape[0], GLOBAL_BATCH_SIZE, validation_steps))
f.write('{}: {}\n{}\n{}\n{}\n'.format(now(), x_train, y_train, x_val, y_val))

# Create in memory shards
f.write('{}: creating shards\n'.format(now()))
from shard import slice_total_gpus, shift_to_rank

_PMI_RANK = int(pmi_rank()) # os env variable
_NNODES = int(pmi_size()) 
#_GPU_NODE = 4
#_TGPUS = _NNODES * _GPU_NODE
#x_train = slice_total_gpus(shift_to_rank(x_train,_PMI_RANK), _TGPUS)
#y_train = slice_total_gpus(shift_to_rank(y_train,_PMI_RANK), _TGPUS)
#x_val = slice_total_gpus(shift_to_rank(x_val,_PMI_RANK), _TGPUS)
#y_val = slice_total_gpus(shift_to_rank(y_val,_PMI_RANK), _TGPUS)
x_train = slice_total_gpus(shift_to_rank(x_train,_PMI_RANK), _NNODES)
y_train = slice_total_gpus(shift_to_rank(y_train,_PMI_RANK), _NNODES)
x_val = slice_total_gpus(shift_to_rank(x_val,_PMI_RANK), _NNODES)
y_val = slice_total_gpus(shift_to_rank(y_val,_PMI_RANK), _NNODES)

f.write('{}: x_train.shape {}\n'.format(now(), x_train.shape))
f.write('{}: y_train.shape {}\n'.format(now(), y_train.shape))

f.write('{}: creating tf.data.Dataset.from_tensor_slices\n'.format(now()))
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(GLOBAL_BATCH_SIZE, 
                                                            drop_remainder=True,
                                                            num_parallel_calls=None,
                                                            deterministic=None,
                                                           ).repeat(EPOCH)
f.write('{}\b'.format(train_ds))

val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(GLOBAL_BATCH_SIZE,
                                                            drop_remainder=True,
                                                            num_parallel_calls=None,
                                                            deterministic=None,).repeat(EPOCH)
f.write('{}\n'.format(val_ds))

options = tf.data.Options()
options.autotune.enabled=True
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
train_ds = train_ds.with_options(options)
val_ds = val_ds.with_options(options)

train_dist = strategy.experimental_distribute_dataset(train_ds)
val_dist = strategy.experimental_distribute_dataset(val_ds)



## Create regression/classifier model using N transformer layers

embed_dim = 128  # Embedding size for each token
num_heads = 16  # Number of attention heads
ff_dim = 128  # Hidden layer size in feed forward network inside transformer

with strategy.scope():
    inputs = layers.Input(shape=(maxlen,))
    embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
    x = embedding_layer(inputs)
    transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
    x = transformer_block(x)
    x = transformer_block(x)
    x = transformer_block(x)
    x = transformer_block(x)
    x = layers.Reshape((1,32000), input_shape=(250,128,))(x)  
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(1024, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(16, activation='relu')(x)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(1, activation='relu')(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.summary()
    model.compile(loss='mean_squared_error',
                  optimizer=Adam(lr=0.00001),
                  metrics=['mae',r2])

## set up a bunch of callbacks to do work during model training..
checkpointer = ModelCheckpoint(
        filepath='smile_regress.autosave.model.h5',
        verbose=1,
        save_weights_only=True,
        save_best_only=True
    )
csv_logger = CSVLogger(
        'smile_regress.training.log'
    )
reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.75,
        patience=20,
        verbose=1,
        mode='auto',
        epsilon=0.0001,
        cooldown=3, 
        min_lr=0.000000001
    )
early_stop = EarlyStopping(monitor='val_loss', patience=50, verbose=1, mode='auto')

f.write('{}: calling model.fit\n'.format(now()))
history = model.fit(
    train_dist,
    batch_size=GLOBAL_BATCH_SIZE,
    steps_per_epoch=int(steps),
    epochs=EPOCH,
    verbose=1,
    validation_data=val_dist,
    validation_steps=validation_steps,
    callbacks = [checkpointer,csv_logger, reduce_lr, early_stop]
)
f.write('{}: done calling model.fit\n'.format(now()))
#history = model.fit(
#        x_train, y_train,
#        batch_size=GLOBAL_BATCH_SIZE,
#        epochs=EPOCH,
#        verbose=1,
#        validation_data=(x_val,y_val),
#        callbacks = [checkpointer,csv_logger, reduce_lr, early_stop]
#        )

#model.load_weights('smile_regress.autosave.model.h5')

