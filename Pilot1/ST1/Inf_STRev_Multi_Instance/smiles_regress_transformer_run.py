############# Module Loading ##############
import argparse
import os
import numpy as np
import matplotlib
import pandas as pd
from mpi4py import MPI


matplotlib.use("Agg")

import tensorflow as tf
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

from clr_callback import *
from smiles_regress_transformer_funcs import *
from tensorflow.python.client import device_lib
import json
from smiles_pair_encoders_functions import *
import time

#######HyperParamSetting#############

json_file = 'config.json'
hyper_params = ParamsJson(json_file)

######## Load model #############

model = ModelArchitecture(hyper_params).call()
model.load_weights(f'smile_regress.autosave.model.h5')

######## Set up MPI #############

comm, size, rank = initialize_mpi()

####### Oranize data files #########

list_dir_files = split_data_list(hyper_params, size, rank)

##### Set up tokenizer ########
if hyper_params['tokenization']['tokenizer']['category'] == 'smilespair':
    vocab_file = hyper_params['tokenization']['tokenizer']['vocab_file']
    spe_file = hyper_params['tokenization']['tokenizer']['spe_file']
    tokenizer = SMILES_SPE_Tokenizer(vocab_file=vocab_file, spe_file= spe_file)

####### Iterate over files ##############
BATCH = hyper_params['general']['batch_size']
start_total = time.time()
preprocess_times = []
inference_times = []

for fil in list_dir_files:

    pp_start = time.time()

    x_inference = inference_data_gen(hyper_params, tokenizer, fil, rank)
    print(len(x_inference))
    pp_end = time.time()
    preprocess_times.append(pp_end - pp_start)

    inf_start = time.time()
    Output = model.predict(x_inference, batch_size = BATCH)
    inf_end = time.time()

    inference_times.append(inf_end - inf_start)
    
    np.savetxt(f'output/{os.path.splitext(fil)[0]}.{rank%4}.dat', np.array(Output).flatten())
    del(Output)
    del(x_inference)

end_total = time.time()

print(f"total time to go through pipeline is {end_total - start_total}")
file1 = open(f"time_info_ranks{size}.batch64.csv", "a")  # append mode
file1.write(f"{rank},{np.mean(preprocess_times)},{np.mean(inference_times)},{end_total - start_total} \n")
file1.close()

