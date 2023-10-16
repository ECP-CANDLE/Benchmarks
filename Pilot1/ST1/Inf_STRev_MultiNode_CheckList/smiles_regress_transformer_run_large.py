############# Module Loading ##############
from collections import OrderedDict
import csv
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
from smiles_regress_transformer_funcs_large import *
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

split_files, split_dirs = large_scale_split(hyper_params, comm, size, rank)

print(len(split_files))

##### Set up tokenizer ########
if hyper_params['tokenization']['tokenizer']['category'] == 'smilespair':
    vocab_file = hyper_params['tokenization']['tokenizer']['vocab_file']
    spe_file = hyper_params['tokenization']['tokenizer']['spe_file']
    tokenizer = SMILES_SPE_Tokenizer(vocab_file=vocab_file, spe_file= spe_file)

####### Iterate over files ##############
BATCH = hyper_params['general']['batch_size']
cutoff = 9
start_total = time.time()
output_dir = hyper_params['general']['output']

for fil, dirs in zip(split_files, split_dirs):

    if True:
        Data_smiles_inf, x_inference = large_inference_data_gen(hyper_params, tokenizer, dirs, fil, rank)

        Output = model.predict(x_inference, batch_size = BATCH)

        '''
        Combine SMILES and predicted docking score.
        Sort the data based on the docking score,
        remove data below cutoff score.
        write data to file in output directory
        '''
        SMILES_DS = np.vstack((Data_smiles_inf, np.array(Output).flatten())).T
        SMILES_DS = sorted(SMILES_DS, key=lambda x: x[1], reverse=True)

        filtered_data = list(OrderedDict((item[0], item) for item in SMILES_DS if item[1] >= cutoff).values())

        filename = f'{output_dir}/{dirs}/{fil}'#{os.path.splitext(fil)[0]}'
        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['smiles', 'score'])
            writer.writerows(filtered_data)

        del (Data_smiles_inf)
        del(Output)
        del(x_inference)
        del(SMILES_DS)
        del(filtered_data)

    #except:
    #    break
        #continue

end_total = time.time()

print(f"total time to go through pipeline is {end_total - start_total}")
file1 = open(f"time_info_ranks{size}.csv", "a")  # append mode
file1.write(f"{rank},{end_total - start_total} \n")
file1.close()

