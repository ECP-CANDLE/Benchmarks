############# Module Loading ##############
import argparse
import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

import horovod.keras as hvd  # ## importing horovod to use data parallelization in another step
import keras_tuner
import tensorflow as tf
from clr_callback import *
from smiles_regress_transformer_funcs_hvd import *
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

#######Argument parsing#############

file_path = os.path.dirname(os.path.realpath(__file__))

# psr and args take input outside of the script and assign:
# (1) file paths for data_path_train and data_path_vali
# (2) number of training epochs

psr = argparse.ArgumentParser(description="input csv file")
psr.add_argument("--in_train", default="in_train")
psr.add_argument("--in_vali", default="in_vali")
psr.add_argument("--ep", type=int, default=400)
psr.add_argument("--num_heads", type=int, default=16)
psr.add_argument("--DR_TB", type=float, default=0.1)
psr.add_argument("--DR_ff", type=float, default=0.1)
psr.add_argument("--activation", default="activation")
psr.add_argument("--drop_post_MHA", type=bool, default=True)
psr.add_argument("--lr", type=float, default=1e-5)
psr.add_argument("--loss_fn", default="mean_squared_error")
psr.add_argument("--hvd_switch", type=bool, default=True)

args = vars(psr.parse_args())  # returns dictionary mapping of an object

######## Set  hyperparameters ########

EPOCH = args["ep"]
num_heads = args["num_heads"]
DR_TB = args["DR_TB"]
DR_ff = args["DR_ff"]
activation = args["activation"]
dropout1 = args["drop_post_MHA"]
lr = args["lr"]
loss_fn = args["loss_fn"]
BATCH = 32  # batch size used for training
vocab_size = 40000
maxlen = 250
# act_fn='elu'
# embed_dim = 128   # Embedding size for each token
# num_heads = 16   # Number of attention heads
# ff_dim = 128   # Hidden layer size in feed forward network inside transformer
checkpt_file = "smile_regress.autosave.model.h5"
csv_file = "smile_regress.training.log"
patience_red_lr = 20
patience_early_stop = 100
hvd_switch = args["hvd_switch"]

########Create training and validation data#####

# x: tokenized sequence data, y: single value dock score
data_path_train = args["in_train"]
data_path_vali = args["in_vali"]

data_train = pd.read_csv(data_path_train)
data_vali = pd.read_csv(data_path_vali)

data_train.head()
# Dataset has type and smiles as the two fields
# reshaping: y formatted as [[y_1],[y_2],...] with floats
y_train = data_train["type"].values.reshape(-1, 1) * 1.0
y_val = data_vali["type"].values.reshape(-1, 1) * 1.0

tokenizer = text.Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(data_train["smiles"])

x_train = prep_text(data_train["smiles"], tokenizer, maxlen)
x_val = prep_text(data_vali["smiles"], tokenizer, maxlen)

######## Implement horovod if necessary ########
if hvd_switch:
    lr, x_train, y_train = initialize_hvd(lr, x_train, y_train)
    x_train, y_train = implement_hvd(x_train, y_train)


######## Build model #############

model = build_model(
    num_heads, DR_TB, DR_ff, activation, dropout1, lr, loss_fn, hvd_switch
)

####### Set callbacks ##############
callbacks = callback_setting(
    hvd_switch, checkpt_file, lr, csv_file, patience_red_lr, patience_early_stop
)

####### Train model! #########

history = model.fit(
    x_train,
    y_train,
    batch_size=BATCH,
    epochs=EPOCH,
    verbose=1,
    validation_data=(x_val, y_val),
    callbacks=callbacks,
)
