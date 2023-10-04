############# Module Loading ##############
import argparse
import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import json

import horovod.keras as hvd  # ## importing horovod to use data parallelization in another step
import tensorflow as tf
from clr_callback import *
from smiles_regress_transformer_spe_funcs import *
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
from tensorflow.python.client import device_lib

#######HyperParamSetting#############

json_file = "config_st_spe_training.json"
hyper_params = ParamsJson(json_file)

if hyper_params["general"]["use_hvd"] == True:
    initialize_hvd()

########Create training and validation data#####
x_train, y_train, x_val, y_val = train_val_data(hyper_params)

######## Build model #############

model = ModelArchitecture(hyper_params).call()

####### Set callbacks + train model ##############

train_and_callbacks = TrainingAndCallbacks(hyper_params)

history = train_and_callbacks.training(
    model, x_train, y_train, (x_val, y_val), hyper_params
)
