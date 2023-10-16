############# Module Loading ##############

import argparse
import os
import numpy as np
import matplotlib
import pandas as pd
import json
from functools import partial

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
import codecs
from SmilesPE.tokenizer import *
#from SmilesPE.spe2vec import *
from smiles_pair_encoders_functions import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import sequence, text
from mpi4py import MPI
from clr_callback import *
from tensorflow.python.client import device_lib
from itertools import chain, repeat, islice

def initialize_hvd():
    hvd.init() 
    print("I am rank %d of %d" %(hvd.rank(), hvd.size()))
    #HVD-2: GPU pinning
    gpus = tf.config.experimental.list_physical_devices('GPU')
    # Ping GPU to each9 rank
    for gpu in gpus:
    	tf.config.experimental.set_memory_growth(gpu,True)
    if gpus:
    	tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
    return 

def initialize_mpi():
    #MPI.Init()
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    return comm, size, rank

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    n_gpus = len([x.name for x in local_device_protos if x.device_type == "GPU"])
    print(f"Num of gpus is {n_gpus}")
    if n_gpus > 1:
        n_gpus -= 1
    
    is_gpu_available = n_gpus > 0
    
    return local_device_protos, [x.name for x in local_device_protos if x.device_type == "GPU"], n_gpus, is_gpu_available


def ParamsJson(json_file):
    with open(json_file) as f:
       params = json.load(f)
    return params

def r2(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - SS_res / (SS_tot + K.epsilon())

def pad_infinite(iterable, padding=None):
   return chain(iterable, repeat(padding))

def pad(iterable, size, padding=None):
   return islice(pad_infinite(iterable, padding), size)

def split_data(data_x, data_y):
    data_x = np.array_split(data_x, hvd.size())[hvd.rank()]
    data_y = np.array_split(data_y, hvd.size())[hvd.rank()]
    return (data_x, data_y)

def prep_text(texts, tokenizer, max_sequence_length):
    # Turns text into into padded sequences.
    text_sequences = tokenizer.texts_to_sequences(texts) # turns text into tokens
    return sequence.pad_sequences(text_sequences, maxlen=max_sequence_length) # pad all sequences so they all have same length

def preprocess_smiles_pair_encoding(data, tokenizer, maxlen):

    tokenized_data = np.array([list(pad(tokenizer(smi)['input_ids'], maxlen, 0)) for smi in data])
    
    return tokenized_data

def train_val_data(hyper_params):

    data_path = hyper_params['data_loading']['data_path']
    rec = hyper_params['data_loading']['rec']
    pattern = hyper_params['data_loading']['pattern']

    tokenizer_params = hyper_params['tokenization']['tokenizer']
    #vocabulary = hyper_params['tokenization']['vocab']
    vocab_size = hyper_params['tokenization']['vocab_size']
    maxlen = hyper_params['tokenization']['maxlen']
    hvd_switch = hyper_params['general']['use_hvd']

    data_train = pd.read_csv(f'{data_path}/ml.{rec}.{pattern}.train')
    data_vali = pd.read_csv(f'{data_path}/ml.{rec}.{pattern}.val')
    
    data_train.head()
    # Dataset has type and smiles as the two fields
    # reshaping: y formatted as [[y_1],[y_2],...] with floats
    x_smiles_train = data_train["smiles"]
    x_smiles_val = data_vali["smiles"]
    y_train = data_train["type"].values.reshape(-1, 1) * 1.0 
    y_val = data_vali["type"].values.reshape(-1, 1) * 1.0

    if hvd_switch:
        x_smiles_train, y_train = split_data(x_smiles_train, y_train)
    
    if tokenizer_params['category'] == 'smilespair':
        spe_file = tokenizer_params['spe_file']
        vocab_file = tokenizer_params['vocab_file']

        tokenizer = SMILES_SPE_Tokenizer(vocab_file=vocab_file, spe_file= spe_file)
        
        x_train = preprocess_smiles_pair_encoding(x_smiles_train,
                                                    tokenizer,
                                                    maxlen
                                                    )

        x_val = preprocess_smiles_pair_encoding(x_smiles_val,
                                                    tokenizer,
                                                    maxlen,
                                                    )

    else:
        tokenizer = text.Tokenizer(num_words=vocab_size)
        tokenizer.fit_on_texts(data_train["smiles"])

        x_train = prep_text(data_train["smiles"], tokenizer, maxlen)
        x_val = prep_text(data_vali["smiles"], tokenizer, maxlen)
    
    ######## Implement horovod if necessary ########
    return x_train, y_train, x_val, y_val


def split_data_list(hyper_params, size, rank):
    DATA_FILE_PATH = hyper_params['inference_data']['data_dir']
    list_dir_files = sorted(os.listdir(DATA_FILE_PATH))
    list_dir_files = np.array_split(np.array(list_dir_files), int(size/4))[int(rank/4)]
    return list_dir_files

def large_scale_split(hyper_params, comm, size, rank):
    output_dir = hyper_params['general']['output']
    checklist_file = hyper_params['general']['checklist']
    restart = hyper_params['general']['restart']
    #print(f"restart is {restart}")
    DATA_FILE_PATH = hyper_params['inference_data']['data_dir']
    databases = hyper_params['inference_data']['databases']
    All_Files = np.array([])
    All_Dirs = np.array([])

    if not restart:
        if rank==0:
            for dirs in databases:
                #if rank==0:
                if not os.path.exists(f'{output_dir}/{dirs}'):
                    try:
                        os.mkdir(f'{output_dir}/{dirs}')
                    except:
                        print("directory exists")
                list_dir_files = np.array(sorted(os.listdir(f'{DATA_FILE_PATH}/{dirs}')))
                All_Files = np.concatenate((All_Files, list_dir_files))
                dir_enumerate = np.array([dirs for i in range(len(list_dir_files))]) 
                All_Dirs = np.concatenate((All_Dirs, dir_enumerate))
                
            checklist = pd.DataFrame({'Directories': All_Dirs, 'Files': All_Files})
            checklist['check'] = [0 for i in range(len(checklist))]
            #if rank==0:
            checklist.to_csv(checklist_file, index=False)

    if restart:
        if rank==0:
            print("Restarting!!!!!!!!!!!")
            checklist = pd.read_csv(checklist_file)
            #checklist_check = checklist[checklist['check']==0]

            for dirs in databases:
                list_dir_files = sorted(os.listdir(f'{output_dir}/{dirs}'))
                #checklist[checklist['Files'].isin(list_dir_files)]['check']=1
                checklist.loc[checklist['Files'].isin(list_dir_files), 'check'] = 1
                print(checklist[checklist['Files'].isin(list_dir_files)])
            checklist.to_csv(checklist_file, index=False) 

            All_Files = np.array(checklist[checklist['check']==0]['Files']) 
            All_Dirs = np.array(checklist[checklist['check']==0]['Directories'])

    All_Files = comm.bcast(All_Files, root = 0)
    All_Dirs = comm.bcast(All_Dirs, root = 0)
    split_files = np.array_split(All_Files, int(size))[int(rank)]
    split_dirs = np.array_split(All_Dirs, int(size))[int(rank)]

    return split_files, split_dirs


def inference_data_gen(hyper_params, tokenizer, fil, rank):
    DATA_FILE_PATH = hyper_params['inference_data']['data_dir']
    data_path_inference = f'{DATA_FILE_PATH}/{fil}'
    maxlen = hyper_params['tokenization']['maxlen']

    Data_smiles_total = pd.read_feather(data_path_inference)['SMILE']
    Data_smiles_raw = np.array_split(Data_smiles_total, 4)[rank%4]
    del(Data_smiles_total)
    
    x_inference = preprocess_smiles_pair_encoding(Data_smiles_raw,
                                                    tokenizer,
                                                    maxlen
                                                    )
    del(Data_smiles_raw)
    
    return x_inference

def _large_inference_data_gen(hyper_params, tokenizer, dirs, fil, rank):
    DATA_FILE_PATH = hyper_params['inference_data']['data_dir']
    data_path_inference = f'{DATA_FILE_PATH}/{dirs}/{fil}'
    maxlen = hyper_params['tokenization']['maxlen']

    Data_smiles_total = pd.read_csv(data_path_inference)['SMILE']
    Data_smiles_raw = np.array_split(Data_smiles_total, 4)[rank%4]
    del(Data_smiles_total)
    
    x_inference = preprocess_smiles_pair_encoding(Data_smiles_raw,
                                                    tokenizer,
                                                    maxlen
                                                    )
    del(Data_smiles_raw)
    
    return x_inference

def large_inference_data_gen(hyper_params, tokenizer, dirs, fil, rank):
    DATA_FILE_PATH = hyper_params['inference_data']['data_dir']
    data_path_inference = f'{DATA_FILE_PATH}/{dirs}/{fil}'
    maxlen = hyper_params['tokenization']['maxlen']

    Data_smiles_total = pd.read_csv(data_path_inference)['SMILE']
    Data_smiles_raw = np.array(Data_smiles_total)#, 4)[rank%4]
    del(Data_smiles_total)

    x_inference = preprocess_smiles_pair_encoding(Data_smiles_raw,
                                                    tokenizer,
                                                    maxlen
                                                    )

    return Data_smiles_raw, x_inference



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

# Implement a Transformer block as a layer
# embed_dim: number of tokens. This is used for the key_dim for the multi-head attention calculation
# ff_dim: number of nodes in Dense layer
# epsilon: needed for numerical stability... not sure what this means to be honest

class TransformerBlock(layers.Layer):
    # __init__: defining all class variables
    def __init__(self, embed_dim, num_heads, ff_dim, rate, activation, dropout1):
        super(TransformerBlock, self).__init__()
        self.drop_chck = dropout1
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)#, activation=activation)
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

class ModelArchitecture(layers.Layer):
    def __init__(self, hyper_params):
                
        lr = hyper_params['general']['lr']
        vocab_size = hyper_params['tokenization']['vocab_size']
        maxlen = hyper_params['tokenization']['maxlen']
        hvd_switch = hyper_params['general']['use_hvd']

        arch_params = hyper_params['architecture']
        embed_dim = arch_params['embedding']['embed_dim']
        num_heads = arch_params['transformer_block']['num_heads']
        ff_dim = arch_params['transformer_block']['ff_dim']
        DR_TB_1 = arch_params['transformer_block']['dr1']
        DR_TB_2 = arch_params['transformer_block']['dr2']
        DR_ff = arch_params['regressor_head']['dr']
        activation_transformer = arch_params['transformer_block']['activation']
        activation_regressor = arch_params['regressor_head']['activation']
        dropout1 = arch_params['transformer_block']['drop_mha']

        self.compile_switch = arch_params['compile']
        self.num_tb = arch_params['transformer_block']['num_blocks']
        self.loss_fn = hyper_params['general']['loss_fn']

        self.inputs = layers.Input(shape=(maxlen,))
        self.embedding_layer = TokenAndPositionEmbedding(maxlen,
                                                        vocab_size,
                                                        embed_dim)

        self.transformer_block = TransformerBlock(embed_dim,
                                                    num_heads,
                                                    ff_dim,
                                                    DR_TB_1,
                                                    activation_transformer,
                                                    dropout1)

        self.reshape = layers.Reshape((1, maxlen * embed_dim),
                                        input_shape=(maxlen, embed_dim,))         

        self.dropout1 = layers.Dropout(DR_ff)
        self.dropout2 = layers.Dropout(DR_ff)
        self.dropout3 = layers.Dropout(DR_ff)
        self.dropout4 = layers.Dropout(DR_ff)
        self.dropout5 = layers.Dropout(DR_ff)

        self.dense1 = layers.Dense(1024, activation=activation_regressor)
        self.dense2 = layers.Dense(256, activation=activation_regressor)
        self.dense3 = layers.Dense(64, activation=activation_regressor)
        self.dense4 = layers.Dense(16, activation=activation_regressor)
        self.dense5 = layers.Dense(1, activation=activation_regressor)

        if hvd_switch:
            lr = lr * hvd.size()
            self.opt = Adam(learning_rate=lr) 
            self.opt = hvd.DistributedOptimizer(self.opt)
        else:
            self.opt = Adam(learning_rate=lr)
    
    def call(self):
        x = self.embedding_layer(self.inputs)
        for tb in range(self.num_tb):
            x = self.transformer_block(x)

        x = self.reshape(x)

        x = self.dropout1(x)
        x = self.dense1(x)

        x = self.dropout2(x)
        x = self.dense2(x)

        x = self.dropout3(x)
        x = self.dense3(x)
        
        x = self.dropout4(x)
        x = self.dense4(x)
        
        x = self.dropout5(x)
        outputs = self.dense5(x)
        
        model = keras.Model(inputs=self.inputs, outputs=outputs)
        model.summary()

        if self.compile_switch:
            model.compile(
                loss=self.loss_fn,
                optimizer=self.opt,
                metrics=["mae", r2],
                steps_per_execution=100
            )
        
        return model

class TrainingAndCallbacks:
    def __init__(self, hyper_params):
        self.hvd_switch = hyper_params['general']['use_hvd']
        checkpt_file = hyper_params['callbacks']['checkpt_file']
        csv_file = hyper_params['callbacks']['log_csv']
        patience_red_lr = hyper_params['callbacks']['patience_red_lr']
        patience_early_stop = hyper_params['callbacks']['patience_early_stop']
        lr = hyper_params['general']['lr']
        if self.hvd_switch:
            lr = lr * hvd.size()

        self.checkpointer = ModelCheckpoint(
            filepath=checkpt_file,
            verbose=1,
            save_weights_only=True,
            save_best_only=True,
            )

        self.clr = CyclicLR(base_lr = lr, max_lr = 5*lr, step_size=2000.)
        self.csv_logger = CSVLogger(csv_file)

        self.reduce_lr = ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.75,
            patience=patience_red_lr,
            verbose=1,
            mode="auto",
            epsilon=0.0001,
            cooldown=3,
            min_lr=0.000000001,
            )

        self.early_stop = EarlyStopping(
            monitor="val_loss",
            patience=patience_early_stop,
            verbose=1,
            mode="auto",
            )

        if self.hvd_switch:
        #HVD broadcast initial variables from rank0 to all other processes 
            self.hvd_broadcast = hvd.callbacks.BroadcastGlobalVariablesCallback(0)

    def callback_defining(self):

        if self.hvd_switch:
            callbacks = [self.hvd_broadcast, self.reduce_lr, self.clr]
            if hvd.rank() == 0:
                callbacks.append(self.csv_logger)
                callbacks.append(self.early_stop)
                callbacks.append(self.checkpointer)
            return callbacks
        else:
            return [self.reduce_lr, self.clr, self.csv_logger, self.early_stop, self.checkpointer]

    def training(self, model, x_train, y_train, validation_data, hyper_params):
        BATCH = hyper_params['general']['batch_size']
        EPOCH = hyper_params['general']['epochs']

        callbacks = self.callback_defining()
        history = model.fit(
            x_train,
            y_train,
            batch_size=BATCH,
            epochs=EPOCH,
            verbose=1,
            validation_data=validation_data,
            callbacks=callbacks,
        )

        return history


