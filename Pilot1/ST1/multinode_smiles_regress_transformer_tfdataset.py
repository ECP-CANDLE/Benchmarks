import os
import sys
import argparse
import numpy as np
import pandas as pd
import json
import logging
logging.basicConfig(
        filename='log',
        level=logging.DEBUG,
        format=os.getenv('PMI_RANK') + ':%(levelname)s:%(message)s'
        )
os.environ['NCCL_DEBUG'] = 'INFO'

import tensorflow as tf
from tensorflow.keras.preprocessing import text
from tensorflow.keras.preprocessing import sequence

from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping

logging.info("starting job")

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


#tf_config = {
#    'cluster': {
#        'worker': ['rbdgx1:12345']
#    },
#    'task': {'type': 'worker', 'index': 0}
#}
from pbsutils import tf_config
os.environ['TF_CONFIG'] = json.dumps(tf_config(port=2223))
num_workers = len(tf_config()['cluster']['worker'])
logging.info(os.environ['TF_CONFIG'])
logging.info(num_workers)

logging.debug('calling tf.distribute.MultiWorkerMirroredStrategy')
strategy = tf.distribute.MultiWorkerMirroredStrategy()
logging.debug('done calling tf.distribute.MultiWorkerMirroredStrategy')

logging.info('tensorflow version: {}'.format(tf.__version__))
logging.info('Number of devices: {}'.format(strategy.num_replicas_in_sync))



file_path = os.path.dirname(os.path.realpath(__file__))
lib_path = os.path.abspath(os.path.join(file_path, '..', '..', 'common'))
sys.path.append(lib_path)



psr = argparse.ArgumentParser(description='input csv file')
psr.add_argument('--x_train',  default='x.train')
psr.add_argument('--x_val',  default='x.val')
psr.add_argument('--y_train', default='y.train')
psr.add_argument('--y_val', default='y.val')
psr.add_argument('--ep',  type=int, default=2)
args=vars(psr.parse_args())
logging.info(args)



EPOCH = args['ep']
#BATCH = 32
#GLOBAL_BATCH_SIZE = BATCH * strategy.num_replicas_in_sync

# TF Tutorial Convention
BATCH_SIZE_PER_REPLICA = 256
GLOBAL_BATCH_SIZE = GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync
logging.info('global batch size: {}, replicas: {}'.format(GLOBAL_BATCH_SIZE, strategy.num_replicas_in_sync))


DR    = 0.1      # Dropout rate



def print_tensor_from_dataset(ds):
    logging.info("\n")
    
    logging.info("\nas tensors:")
    
    for element in ds.take(1):
        logging.info (element)

    logging.info("\nas numpy:")

    for element in ds.take(1).as_numpy_iterator():
        logging.info (element)
    
    logging.info("\n")


x_train = tf.data.TextLineDataset(args["x_train"], name="x_train")
x_val = tf.data.TextLineDataset(args["x_val"], name="x_val")

y_train = tf.data.TextLineDataset(args["y_train"], name="y_train")
y_val = tf.data.TextLineDataset(args["y_val"], name="y_val")

# TextLineDataset reads numbers from file as text

y_train = y_train.map(lambda x: float(x), num_parallel_calls=tf.data.AUTOTUNE)
y_val = y_val.map(lambda x: float(x), num_parallel_calls=tf.data.AUTOTUNE)

# transform y from [value1, value2] to [[value1], [value2]]
y_train = y_train.map(lambda element: tf.expand_dims(element, axis=0), num_parallel_calls=tf.data.AUTOTUNE)
y_val = y_val.map(lambda element: tf.expand_dims(element, axis=0), num_parallel_calls=tf.data.AUTOTUNE)



vocab_size = 40000  # 
maxlen = 250  # 

vectorize_layer = tf.keras.layers.TextVectorization(
    output_mode='int',
    standardize=None,
    max_tokens=vocab_size,
    split='character',
    output_sequence_length=maxlen,
    pad_to_max_tokens=True
)

# alternative to passing in a precomputed vocabulary
# might want to use a precomputed vocabulary with large dataset

batch_size=1000
steps=1600000/batch_size
vectorize_layer.adapt(x_train, batch_size=batch_size, steps=steps)

x_train = x_train.map(vectorize_layer, num_parallel_calls=tf.data.AUTOTUNE)
x_val = x_val.map(vectorize_layer, num_parallel_calls=tf.data.AUTOTUNE)

# reshape for input layer
x_train = x_train.map(lambda x: tf.reshape(x, [1,250]), num_parallel_calls=tf.data.AUTOTUNE)
y_train = y_train.map(lambda x: tf.reshape(x, [1,1]), num_parallel_calls=tf.data.AUTOTUNE)
x_val = x_val.map(lambda x: tf.reshape(x, [1,250]), num_parallel_calls=tf.data.AUTOTUNE)
y_val = y_val.map(lambda x: tf.reshape(x, [1,1]), num_parallel_calls=tf.data.AUTOTUNE)

logging.info('x_train')
#logging.info_tensor_from_dataset(x_train)
logging.info('y_train')
#logging.info_tensor_from_dataset(y_train)
logging.info('x_val')
#logging.info_tensor_from_dataset(x_val)
logging.info('y_val')
#print_tensor_from_dataset(y_val)


# define r2 for reporting
def r2(y_true, y_pred):
    SS_res =  K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return (1 - SS_res/(SS_tot + K.epsilon()))

# Implement a Transformer block as a layer
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



# Create regression/classifier model using N transformer layers
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

    #x = layers.GlobalAveragePooling1D()(x)  --- the original model used this but the accuracy was much lower

    x = layers.Reshape((1,32000), input_shape=(250,128,))(x)  # reshaping increases parameters but improves accuracy a lot
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



    # Train and Evaluate

    model.compile(loss='mean_squared_error',
              optimizer=Adam(lr=0.00001),
              metrics=['mae',r2])



# set up a bunch of callbacks to do work during model training..

checkpointer = ModelCheckpoint(filepath='smile_regress.autosave.model.h5', verbose=1, save_weights_only=True, save_best_only=True)
csv_logger = CSVLogger('smile_regress.training.log')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.75, patience=20, verbose=1, mode='auto', epsilon=0.0001, cooldown=3, min_lr=0.000000001)
early_stop = EarlyStopping(monitor='val_loss', patience=50, verbose=1, mode='auto')


tensorboard = tf.keras.callbacks.TensorBoard(log_dir='logs', histogram_freq=1)



train=tf.data.Dataset.zip((x_train, y_train))
val = tf.data.Dataset.zip((x_val, y_val))

train.prefetch(GLOBAL_BATCH_SIZE)
val.prefetch(GLOBAL_BATCH_SIZE)

train.batch(GLOBAL_BATCH_SIZE)
val.batch(GLOBAL_BATCH_SIZE)

train_dist_dataset = strategy.experimental_distribute_dataset(train)
test_dist_dataset = strategy.experimental_distribute_dataset(val)

options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
train_dist = dataset.with_options(options)
val_dist = dataset.with_options(options)

train_dist.cache()
val_dist.cache()

#print_tensor_from_dataset(train)
#print_tensor_from_dataset(val)



history = model.fit(train,
                    batch_size=GLOBAL_BATCH_SIZE,
                    steps_per_epoch=1600000/GLOBAL_BATCH_SIZE,
                    validation_batch_size=10000,
                    validation_freq=2,
                    epochs=EPOCH,
                    verbose=1,
                    validation_data=val,
                    callbacks = [checkpointer,csv_logger, reduce_lr, early_stop, tensorboard])

model.load_weights('smile_regress.autosave.model.h5')

