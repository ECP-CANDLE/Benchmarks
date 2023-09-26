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
import horovod.keras as hvd ### importing horovod to use data parallelization in another step
from clr_callback import *
import deephyper
from deephyper.problem import HpProblem
from tensorflow.python.client import device_lib
import ray
from deephyper.evaluator import Evaluator
from deephyper.evaluator.callback import TqdmCallback
from deephyper.search.hps import CBO
from itertools import chain, repeat, islice

def pad_infinite(iterable, padding=None):
   return chain(iterable, repeat(padding))

def pad(iterable, size, padding=None):
   return islice(pad_infinite(iterable, padding), size)


def ParamsJson(json_file):
    with open(json_file) as f:
       params = json.load(f)
    return params


def ArgParsing():
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
    
    args = vars(psr.parse_args()) # returns dictionary mapping of an object
    
    ######## Set  hyperparameters ########
    data_path_train = args["in_train"]
    data_path_vali = args["in_vali"]

    EPOCH = args["ep"]
    num_heads = args["num_heads"]
    DR_TB = args["DR_TB"]
    DR_ff = args["DR_ff"]
    activation = args["activation"]
    dropout1 = args["drop_post_MHA"]
    lr = args["lr"]
    loss_fn = args["loss_fn"]
    hvd_switch = args["hvd_switch"]

    return data_path_train, data_path_vali, EPOCH, num_heads, DR_TB, DR_ff, activation, dropout1, lr, loss_fn, hvd_switch

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


def split_data(data_x, data_y):
    data_x = np.array_split(data_x, hvd.size())[hvd.rank()]
    data_y = np.array_split(data_y, hvd.size())[hvd.rank()]
    return (data_x, data_y)


#def implement_hvd(x_train, y_train):
#    x_train = x_train[hvd.rank()]
#    y_train = y_train[hvd.rank()]
#    return (x_train, y_train)

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
    text_sequences = tokenizer.texts_to_sequences(texts) # turns text into tokens
    return sequence.pad_sequences(text_sequences, maxlen=max_sequence_length) # pad all sequences so they all have same length


#def train_val_data(data_path_train, data_path_vali, hvd_switch, vocab_size, maxlen):

def preprocess_smiles_pair_encoding(data, maxlen, vocab_file, spe_file):
    # some default tokens from huggingface
    default_toks = ['[PAD]', 
                    '[unused1]', '[unused2]', '[unused3]', '[unused4]','[unused5]', '[unused6]', '[unused7]', '[unused8]', '[unused9]', '[unused10]', 
                    '[UNK]', '[CLS]', '[SEP]', '[MASK]']
    
    
    # atom-level tokens used for trained the spe vocabulary
    atom_toks = ['[c-]', '[SeH]', '[N]', '[C@@]', '[Te]', '[OH+]', 'n', '[AsH]', '[B]', 'b', 
                 '[S@@]', 'o', ')', '[NH+]', '[SH]', 'O', 'I', '[C@]', '-', '[As+]', '[Cl+2]', 
                 '[P+]', '[o+]', '[C]', '[C@H]', '[CH2]', '\\', 'P', '[O-]', '[NH-]', '[S@@+]', 
                 '[te]', '[s+]', 's', '[B-]', 'B', 'F', '=', '[te+]', '[H]', '[C@@H]', '[Na]', 
                 '[Si]', '[CH2-]', '[S@+]', 'C', '[se+]', '[cH-]', '6', 'N', '[IH2]', '[As]', 
                 '[Si@]', '[BH3-]', '[Se]', 'Br', '[C+]', '[I+3]', '[b-]', '[P@+]', '[SH2]', '[I+2]', 
                 '%11', '[Ag-3]', '[O]', '9', 'c', '[N-]', '[BH-]', '4', '[N@+]', '[SiH]', '[Cl+3]', '#', 
                 '(', '[O+]', '[S-]', '[Br+2]', '[nH]', '[N+]', '[n-]', '3', '[Se+]', '[P@@]', '[Zn]', '2', 
                 '[NH2+]', '%10', '[SiH2]', '[nH+]', '[Si@@]', '[P@@+]', '/', '1', '[c+]', '[S@]', '[S+]', 
                 '[SH+]', '[B@@-]', '8', '[B@-]', '[C-]', '7', '[P@]', '[se]', 'S', '[n+]', '[PH]', '[I+]', '5', 'p', '[BH2-]', '[N@@+]', '[CH]', 'Cl']
    
    tokenizer = SMILES_SPE_Tokenizer(vocab_file=vocab_file, spe_file= spe_file)

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
        x_train = preprocess_smiles_pair_encoding(x_smiles_train,
                                                    maxlen,
                                                    vocab_file,
                                                    spe_file)

        x_val = preprocess_smiles_pair_encoding(x_smiles_val,
                                                    maxlen,
                                                    vocab_file,
                                                    spe_file)
        print(x_train)

    else:
        tokenizer = text.Tokenizer(num_words=vocab_size)
        tokenizer.fit_on_texts(data_train["smiles"])

        x_train = prep_text(data_train["smiles"], tokenizer, maxlen)
        x_val = prep_text(data_vali["smiles"], tokenizer, maxlen)
    
    ######## Implement horovod if necessary ########
    #if hvd_switch:
    #    x_train, y_train = initialize_hvd(x_train, y_train)
    #    x_train, y_train = implement_hvd(x_train, y_train)

    return x_train, y_train, x_val, y_val

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    n_gpus = len([x.name for x in local_device_protos if x.device_type == "GPU"])
    print(f"Num of gpus is {n_gpus}")
    if n_gpus > 1:
        n_gpus -= 1
    
    is_gpu_available = n_gpus > 0
    
    #if is_gpu_available:
        #print(f"{n_gpus} GPU{'s are' if n_gpus > 1 else ' is'} available.")
    #else:
        #print("No GPU available")

    return local_device_protos, [x.name for x in local_device_protos if x.device_type == "GPU"], n_gpus, is_gpu_available


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
    #def __init__(self, vocab_size, maxlen, embed_dim, num_heads, ff_dim, DR_TB, DR_ff, activation, dropout1, lr, loss_fn, hvd_switch):
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

        model.compile(
            loss=self.loss_fn, optimizer=self.opt, metrics=["mae", r2], steps_per_execution=100
        )
        
        return model

class TrainingAndCallbacks:
    #def __init__(self, hvd_switch, checkpt_file, lr, csv_file, patience_red_lr, patience_early_stop):
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
            filepath=checkpt_file,#"smile_regress.autosave.model.h5",
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



class RunFnDeepHyper:
    def __init__(self, x_train, y_train, x_val, y_val): 
        # Params that are currently static
        self.vocab_size = 40000
        self.maxlen = 250
        self.embed_dim = 128
        self.ff_dim = 128
        self.BATCH = 32
        self.patience_red_lr = 20
        self.patience_early_stop = 100
        self.hvd_switch = False
        self.checkpt_file = 'smile_regress.autosave.model.h5'
        self.csv_file = 'smile_regress.training.log'

        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val

    def run(self, config):
        
        num_heads = config["num_heads"]
        DR_TB = config["DR_TB"]
        DR_ff = config["DR_ff"]
        activation = config["activation"]
        dropout1 = config["dropout_aftermulti"]
        lr = config["lr"]
        loss_fn = config["loss_fn"]
        EPOCH = config["epochs"]
        validation_data = (self.x_val, self.y_val)

        model = ModelArchitecture(self.vocab_size, self.maxlen, self.embed_dim, num_heads, self.ff_dim, DR_TB, DR_ff, activation, dropout1, lr, loss_fn, self.hvd_switch).call()

        history = TrainingAndCallbacks(self.hvd_switch, self.checkpt_file, lr, self.csv_file, self.patience_red_lr, self.patience_early_stop).training( model, self.x_train, self.y_train, validation_data, self.BATCH, EPOCH)

        return history.history["val_accuracy"] [-1]


def run(config):

    DATA_PATH='/grand/datascience/avasan/ST_Benchmarks/Data/1M-flatten'
    
    TFIL='ml.3CLPro_7BQY_A_1_F.Orderable_zinc_db_enaHLL.sorted.4col.dd.parquet.xform-smiles.csv.reg.train'
    
    VFIL='ml.3CLPro_7BQY_A_1_F.Orderable_zinc_db_enaHLL.sorted.4col.dd.parquet.xform-smiles.csv.reg.val'
    
    data_path_train = f'{DATA_PATH}/{TFIL}'
    data_path_vali = f'{DATA_PATH}/{TFIL}'
    hvd_switch = False
    BATCH = 32 # batch size used for training
    vocab_size = 40000
    maxlen = 250
    embed_dim = 128   # Embedding size for each token
    ff_dim = 128   # Hidden layer size in feed forward network inside transformer
    checkpt_file = "smile_regress.autosave.model.h5"
    csv_file = "smile_regress.training.log"
    patience_red_lr = 20
    patience_early_stop = 100
    
    ########Create training and validation data##### 
    x_train, y_train, x_val, y_val = train_val_data(data_path_train, data_path_vali, hvd_switch, vocab_size, maxlen)
    num_heads = config["num_heads"]
    DR_TB = config["DR_TB"]
    DR_ff = config["DR_ff"]
    activation = config["activation"]
    dropout1 = config["dropout_aftermulti"]
    lr = config["lr"]
    loss_fn = config["loss_fn"]
    EPOCH = config["epochs"]
    validation_data = (x_val, y_val)

    model = ModelArchitecture(vocab_size, maxlen, embed_dim, num_heads, ff_dim, DR_TB, DR_ff, activation, dropout1, lr, loss_fn, hvd_switch).call()

    history = TrainingAndCallbacks(hvd_switch, checkpt_file, lr, csv_file, patience_red_lr, patience_early_stop).training( model, x_train, y_train, validation_data, BATCH, EPOCH)

    return history.history["val_accuracy"] [-1]



def hyper_param_problem():

    ACTIVATIONS = [
    "elu", "gelu", "hard_sigmoid", "linear", "relu", "selu",
    "sigmoid", "softplus", "softsign", "swish", "tanh",
    ]

    LRS = [1e-6 * i for i in range(0,11)]
    
    LOSSFNS = ["mean_squared_error", "mean_absolute_error"]

    problem = HpProblem()
    problem.add_hyperparameter((12, 32), "num_heads", default_value = 16)
    problem.add_hyperparameter((0.025, 0.5), "DR_TB", default_value = 0.1)
    problem.add_hyperparameter((0.025, 0.5), "DR_ff", default_value = 0.1)
    problem.add_hyperparameter(ACTIVATIONS, "activation", default_value = "elu")
    problem.add_hyperparameter((1e-7, 1e-5), "lr", default_value = 1e-6)
    problem.add_hyperparameter(LOSSFNS, "loss_fn", default_value = "mean_squared_error")
    problem.add_hyperparameter((2,10), "epochs", default_value = 2)
    problem.add_hyperparameter([True, False], "dropout_aftermulti", default_value = False)

    return problem


def default_evaluation(problem, is_gpu_available, n_gpus, run):
    if is_gpu_available:
        if not(ray.is_initialized()):
            ray.init(num_cpus=n_gpus, num_gpus=n_gpus, log_to_driver=False)
    

        run_default = ray.remote(num_cpus=1, num_gpus=1)(run)
        objective_default = ray.get(run_default.remote(problem.default_configuration))
    else:
        if not(ray.is_initialized()):
            ray.init(num_cpus=1, log_to_driver=False)
        run_default = run
        print(problem.default_configuration)
        objective_default = run_default(problem.default_configuration)
    return objective_default


def get_evaluator(run_function, is_gpu_available, n_gpus):
    # Default arguments for Ray: 1 worker and 1 worker per evaluation
    method_kwargs = {
        "num_cpus": 1,
        "num_cpus_per_task": 1,
        "callbacks": [TqdmCallback()]
    }

    # If GPU devices are detected then it will create 'n_gpus' workers
    # and use 1 worker for each evaluation
    if is_gpu_available:
        method_kwargs["num_cpus"] = n_gpus
        method_kwargs["num_gpus"] = n_gpus
        method_kwargs["num_cpus_per_task"] = 1
        method_kwargs["num_gpus_per_task"] = 1

    evaluator = Evaluator.create(
        run_function,
        method="ray",
        method_kwargs=method_kwargs
    )
    print(f"Created new evaluator with {evaluator.num_workers} worker{'s' if evaluator.num_workers > 1 else ''} and config: {method_kwargs}", )

    return evaluator


def build_model_tuner(hp):
    #units = hp.Int("units", min_value=32, max_value=512, step=32)
    vocab_size = 40000
    maxlen = 250
    embed_dim = 128
    num_heads = hp.Int("num_heads", min_value=12, max_value=40, step=4)
    ff_dim = 128
    DR_TB = hp.Float("DR_TB", min_value=0.025, max_value=0.5, step=0.025)
    DR_ff = hp.Float("DR_TB", min_value=0.025, max_value=0.5, step=0.025)
    activation = hp.Choice("activation", ["relu", "elu", "gelu"])
    #activation="elu"
    dropout1 = hp.Boolean("dropout_aftermulti")
    lr = hp.Float("lr", min_value=1e-6, max_value=1e-5, step=1e-6)
    loss_fn = hp.Choice("loss_fn", ["mean_squared_error", "mean_absolute_error"])
    # call existing model-building code with the hyperparameter values.
    model = ModelArchitecture(vocab_size, maxlen, embed_dim, num_heads, ff_dim, DR_TB, DR_ff, activation, dropout1, lr, loss_fn, hvd_switch).call()

    return model


#tfm.optimization.lars_optimizer.LARS(
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

def model_architecture(embed_dim, num_heads, ff_dim, DR_TB, DR_ff, activation, dropout1, lr, loss_fn, hvd_switch):

    vocab_size = 40000  #number of possible 'words' in SMILES data
    maxlen = 250  #length of each SMILE sequence in input
    inputs = layers.Input(shape=(maxlen,))
    embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
    x = embedding_layer(inputs)
    transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim, DR_TB, activation, dropout1)
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
    
    #HVD Wrap optimizer in hvd Distributed Optimizer delegates gradient comp to original optimizer, averages gradients, and applies averaged gradients
    if hvd_switch:
        opt = hvd.DistributedOptimizer(opt)

    model.compile(
        loss=loss_fn, optimizer=opt, metrics=["mae", r2]
    )
    return model

def callback_setting(hvd_switch, checkpt_file, lr, csv_file, patience_red_lr, patience_early_stop):
    
    checkpointer = ModelCheckpoint(
        filepath=checkpt_file,#"smile_regress.autosave.model.h5",
        verbose=1,
        save_weights_only=True,
        save_best_only=True,
    )
    
    clr = CyclicLR(base_lr = lr, max_lr = 5*lr, step_size=2000.)
    
    csv_logger = CSVLogger(csv_file)#"smile_regress.training.log")
    
    # learning rate tuning at each epoch
    # is it possible to do batch size tuning at each epoch as well? 
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.75,
        patience=patience_red_lr,#20,
        verbose=1,
        mode="auto",
        epsilon=0.0001,
        cooldown=3,
        min_lr=0.000000001,
    )

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=patience_early_stop,#100,
        verbose=1,
        mode="auto",
        )

    if hvd_switch:
        #HVD broadcast initial variables from rank0 to all other processes 
        hvd_broadcast = hvd.callbacks.BroadcastGlobalVariablesCallback(0)

        callbacks = [hvd_broadcast,reduce_lr,clr]

        if hvd.rank() == 0:
            callbacks.append(csv_logger)
            callbacks.append(early_stop)
            callbacks.append(checkpointer)

        return callbacks

    else:
        return [reduce_lr, clr, csv_logger, early_stop, checkpointer]


def build_model_DeepHyper(x_train, y_train, x_val, y_val, config, hvd_switch=False, checkpt_file = 'smile_regress.autosave.model.h5', csv_file = 'smile_regress.training.log'):
    #units = hp.Int("units", min_value=32, max_value=512, step=32)
    embed_dim = 128
    ff_dim = 128
    BATCH = 32
    patience_red_lr = 20
    patience_early_stop = 100

    num_heads = config["num_heads"]
    DR_TB = config["DR_TB"]
    DR_ff = config["DR_ff"]
    activation = config["activation"]
    dropout1 = config["dropout_aftermulti"]
    lr = config["lr"]
    loss_fn = config["loss_fn"]
    EPOCH = config["epochs"]

        # call existing model-building code with the hyperparameter values.
    model = model_architecture (
        embed_dim=embed_dim, num_heads=num_heads, ff_dim=ff_dim, DR_TB=DR_TB, DR_ff = DR_ff, activation=activation, dropout1=dropout1, lr=lr, loss_fn=loss_fn
    )

    callbacks = callback_setting (
        hvd_switch,
        checkpt_file,
        lr,
        csv_file,
        patience_red_lr,
        patience_early_stop
        )
    
    history = model.fit(
        x_train,
        y_train,
        batch_size=BATCH,
        epochs=EPOCH,
        verbose=1,
        validation_data=(x_val, y_val),
        callbacks=callbacks,
    )

    return history.history["val_accuracy"] [-1]




#def build_model(num_heads, DR_TB, DR_ff, activation, dropout1, lr, loss_fn, hvd_switch):
#    #units = hp.Int("units", min_value=32, max_value=512, step=32)
#    embed_dim = 128
#    ff_dim = 128
#    # call existing model-building code with the hyperparameter values.
#    model = model_architecture (
#        embed_dim=embed_dim, num_heads=num_heads, ff_dim=ff_dim, DR_TB=DR_TB, DR_ff = DR_ff, activation=activation, dropout1=dropout1, lr=lr, loss_fn=loss_fn, hvd_switch=hvd_switch
#    )
#    return model



