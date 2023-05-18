# Setup

import pandas as pd
# import numpy as np
import os
import sys
# import gzip
import argparse

# import math
import matplotlib
matplotlib.use('Agg')

# import matplotlib.pyplot as plt

import tensorflow as tf
options = tf.profiler.experimental.ProfilerOptions(host_tracer_level=3, python_tracer_level=1, device_tracer_level=1)
import time

from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping, TensorBoard
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing import text

file_path = os.path.dirname(os.path.realpath(__file__))
lib_path = os.path.abspath(os.path.join(file_path, '..', '..', 'common'))
sys.path.append(lib_path)

psr = argparse.ArgumentParser(description='input csv file')
psr.add_argument('--in_train', default='in_train')
psr.add_argument('--in_vali', default='in_vali')
psr.add_argument('--ep', type=int, default=400)
psr.add_argument('--profiler', type=bool, default=False)
psr.add_argument('--prof_dir', default="profile_data")
psr.add_argument('--batch_log', default=False)

args = vars(psr.parse_args())
print(args)

EPOCH = args['ep']
BATCH = 32

data_path_train = args['in_train']
data_path_vali = args['in_vali']

DR = 0.1      # Dropout rate


class MyCallBack(keras.callbacks.Callback):
    def __init__(self, args):
        super().__init__()
        self.batchsize = BATCH
        self.logfreq = 10
        self.batch_begin_time = 0
        self.batch_end_time = 0
        self.max_speed = 0
        self.epoch_time = 0
        self.train_time = 0
        self.args = args

    def on_batch_begin(self, batch, logs=None):
        if batch == 10 and self.args['profiler'] is not None and self.args['profiler'] is True:
            tf.profiler.experimental.start(self.args['prof_dir'], options=options)
        if batch % self.logfreq == 0:
            self.batch_begin_time = time.time()
        self.batch_begin_time = time.time()

    def on_batch_end(self, batch, logs=None):
        self.epoch_batch_count += 1
        self.train_batch_count += 1
        self.batch_time = time.time() - self.batch_begin_time
        self.epoch_time += self.batch_time
        if batch == 10 and self.args['profiler'] is not None and self.args['profiler'] is True:
            tf.profiler.experimental.stop()
            sys.exit()

        if batch % self.logfreq == 0:
            self.batch_speed = self.batchsize / self.batch_time
            if self.batch_speed > self.max_speed:
                self.max_speed = self.batch_speed
        print(f"batch {batch} time(s) {round(self.batch_time, 6)} throughput(samples/sec): {round(self.batch_speed, 3)}")

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_batch_count = 0
        self.epoch_begin_time = time.time()
        self.epoch_time = 0

    def on_epoch_end(self, epoch, logs=None):
        self.train_time += self.epoch_time
        self.epoch_avg_speed = self.epoch_batch_count * self.batchsize / self.epoch_time
        print(f"epoch {epoch} time (s):", round(self.epoch_time, 3), " throughput(samples/sec):", round(self.epoch_avg_speed, 3))

    def on_train_begin(self, logs=None):
        self.train_batch_count = 0
        self.train_begin_time = time.time()
        self.train_time = 0

    def on_train_end(self, logs=None):
        speed_train = (self.batchsize * self.train_batch_count) / self.train_time
        print("Total train time(s) :", round(self.train_time, 3), " batches:", self.train_batch_count, " batchsize:",  self.batchsize, " throughput(samples/sec) (avg, max): ", round(speed_train, 3), round(self.max_speed, 3))


# define r2 for reporting


def r2(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return (1 - SS_res / (SS_tot + K.epsilon()))

# Implement a Transformer block as a layer


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim), ]
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


# Input and prepare dataset

vocab_size = 40000  #
maxlen = 250  #


data_train = pd.read_csv(data_path_train)
data_vali = pd.read_csv(data_path_vali)

data_train.head()

# Dataset has type and smiles as the two fields

y_train = data_train["type"].values.reshape(-1, 1) * 1.0
y_val = data_vali["type"].values.reshape(-1, 1) * 1.0

tokenizer = text.Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(data_train["smiles"])


def prep_text(texts, tokenizer, max_sequence_length):
    # Turns text into into padded sequences.
    text_sequences = tokenizer.texts_to_sequences(texts)
    return sequence.pad_sequences(text_sequences, maxlen=maxlen)


x_train = prep_text(data_train["smiles"], tokenizer, maxlen)
x_val = prep_text(data_vali["smiles"], tokenizer, maxlen)

print(x_train.shape)
print(y_train.shape)

# Create regression/classifier model using N transformer layers

embed_dim = 128  # Embedding size for each token
num_heads = 16  # Number of attention heads
ff_dim = 128  # Hidden layer size in feed forward network inside transformer

inputs = layers.Input(shape=(maxlen,))
embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
x = embedding_layer(inputs)
transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
x = transformer_block(x)
x = transformer_block(x)
x = transformer_block(x)
x = transformer_block(x)

# x = layers.GlobalAveragePooling1D()(x)  --- the original model used this but the accuracy was much lower

x = layers.Reshape((1, 32000), input_shape=(250, 128,))(x)  # reshaping increases parameters but improves accuracy a lot
x = layers.Dropout(0.1)(x)
x = layers.Dense(1024, activation="relu")(x)
x = layers.Dropout(0.1)(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.1)(x)
x = layers.Dense(64, activation="relu")(x)
x = layers.Dropout(0.1)(x)
x = layers.Dense(16, activation="relu")(x)
x = layers.Dropout(0.1)(x)
outputs = layers.Dense(2, activation="softmax")(x)

model = keras.Model(inputs=inputs, outputs=outputs)

model.summary()

# Train and Evaluate

model.compile(loss='sparse_categorical_crossentropy',
              optimizer=Adam(lr=0.000001),
              metrics=['accuracy'])

# set up a bunch of callbacks to do work during model training..

checkpointer = ModelCheckpoint(filepath='smile_class.autosave.model.h5', verbose=1, save_weights_only=True, save_best_only=True)
csv_logger = CSVLogger('smile_class.training.log')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.75, patience=20, verbose=1, mode='auto', epsilon=0.0001, cooldown=3, min_lr=0.000000001)
early_stop = EarlyStopping(monitor='val_loss', patience=100, verbose=1, mode='auto')

callbacks = [checkpointer, csv_logger, reduce_lr, early_stop]

if args['profiler'] is not None and args['profiler'] is True:
    tensorboard = TensorBoard(log_dir=args['prof_dir'])
    callbacks.append(tensorboard)

my_hook = MyCallBack(args)
callbacks.append(my_hook)

history = model.fit(x_train, y_train,
                    batch_size=BATCH,
                    epochs=EPOCH,
                    verbose=1,
                    validation_data=(x_val, y_val),
                    callbacks=callbacks)

model.load_weights('smile_class.autosave.model.h5')
