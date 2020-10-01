from __future__ import print_function

import os
import sys
import logging


file_path = os.path.dirname(os.path.realpath(__file__))
lib_path = os.path.abspath(os.path.join(file_path, '..'))
sys.path.append(lib_path)
lib_path2 = os.path.abspath(os.path.join(file_path, '..', '..', 'common'))
sys.path.append(lib_path2)

import candle

logger = logging.getLogger(__name__)
# candle.set_parallelism_threads()

additional_definitions = [
    # Data set selection
    {'name': 'train_sources',
        'default': 'gCSI',
        'choices': ['CCLE', 'CTRP', 'gCSI', 'GDSC', 'NCI60'],
        'help':'use one source of drug response data for training'},
    {'name': 'val_sources',
        'nargs': '+',
        'choices': ['train', 'all', 'CCLE', 'CTRP', 'gCSI', 'GDSC', 'NCI60'],
        'help':'use one or more sources of drug response data for validation'},
    # Pre-processing for dataframes
    {'name': 'grth_scaling',
        'default': 'std',
        'choices': ['minmax', 'std', 'none'],
        'help':'type of scaling method for drug response (growth); "minmax": to [0,1], "std": standard unit normalization; "none": no normalization'},
    {'name': 'dscptr_scaling',
        'default': 'std',
        'choices': ['minmax', 'std', 'none'],
        'help':'type of scaling method for drug feature (descriptor); "minmax": to [0,1], "std": standard unit normalization; "none": no normalization'},
    {'name': 'rnaseq_scaling',
        'default': 'std',
        'choices': ['minmax', 'std', 'none'],
        'help':'type of scaling method for RNA sequence; "minmax": to [0,1], "std": standard unit normalization; "none": no normalization'},
    {'name': 'qed_scaling',
        'default': 'none',
        'choices': ['minmax', 'std', 'none'],
        'help':'type of scaling method for drug weighted QED; "minmax": to [0,1], "std": standard unit normalization; "none": no normalization'},
    {'name': 'dscptr_nan_threshold',
        'type': float,
        'default': 0.0,
        'help': 'ratio of NaN values allowed for drug descriptor'},
    # Feature usage and partitioning settings
    {'name': 'rnaseq_feature_usage',
        'choices': ['source_scale', 'combat'],
        'default':'combat',
        'help':'RNA sequence data used'},
    {'name': 'drug_feature_usage',
        'choices': ['fingerprint', 'descriptor', 'both'],
        'default':'both',
        'help':'drug features (fp and/or desc) used'},
    {'name': 'disjoint_drugs',
        'type': candle.str2bool,
        'default': False,
        'help': 'disjoint drugs between train/validation'},
    {'name': 'disjoint_cells',
        'type': candle.str2bool,
        'default': False,
        'help': 'disjoint_cells between train/validation'},
    # Network configuration ###################################################
    # Encoders for drug features and RNA sequence (LINCS 1000)
    {'name': 'gene_layer_dim',
        'default': 1024,
        'type': int,
        'help': 'dimension of layers for RNA sequence'},
    {'name': 'gene_latent_dim',
        'default': 256,
        'type': int,
        'help': 'dimension of latent variable for RNA sequence'},
    {'name': 'gene_num_layers',
        'default': 2,
        'type': int,
        'help': 'number of layers for RNA sequence'},
    {'name': 'drug_layer_dim',
        'default': 4096,
        'type': int,
        'help': 'dimension of layers for drug feature'},
    {'name': 'drug_latent_dim',
        'default': 1024,
        'type': int,
        'help': 'dimension of latent variable for drug feature'},
    {'name': 'drug_num_layers',
        'default': 2,
        'type': int,
        'help': 'number of layers for drug feature'},
    # Using autoencoder for drug/sequence encoder initialization
    {'name': 'autoencoder_init',
        'type': candle.str2bool,
        'default': False,
        'help': 'indicator of autoencoder initialization for drug/RNA sequence feature encoder'},
    # Drug response regression network
    {'name': 'resp_layer_dim',
        'default': 1024,
        'type': int,
        'help': 'dimension of layers for drug response block'},
    {'name': 'resp_num_layers_per_block',
        'default': 2,
        'type': int,
        'help': 'number of layers for drug response res block'},
    {'name': 'resp_num_blocks',
        'default': 2,
        'type': int,
        'help': 'number of residual blocks for drug response'},
    {'name': 'resp_num_layers',
        'default': 2,
        'type': int,
        'help': 'number of layers for drug response'},
    {'name': 'resp_activation',
        'default': 'none',
        'choices': ['sigmoid', 'tanh', 'none'],
        'help':'activation for response prediction output'},
    # Cell line classification network(s)
    {'name': 'cl_clf_layer_dim',
        'default': 256,
        'type': int,
        'help': 'layer dimension for cell line classification'},
    {'name': 'cl_clf_num_layers',
        'default': 1,
        'type': int,
        'help': 'number of layers for cell line classification'},
    # Drug target family classification network
    {'name': 'drug_target_layer_dim',
        'default': 512,
        'type': int,
        'help': 'dimension of layers for drug target prediction'},
    {'name': 'drug_target_num_layers',
        'default': 2,
        'type': int,
        'help': 'number of layers for drug target prediction'},
    # Drug weighted QED regression network
    {'name': 'drug_qed_layer_dim',
        'default': 512,
        'type': int,
        'help': 'dimension of layers for QED prediction'},
    {'name': 'drug_qed_num_layers',
        'default': 2,
        'type': int,
        'help': 'number of layers for QED prediction'},
    {'name': 'drug_qed_activation',
        'default': 'none',
        'choices': ['sigmoid', 'tanh', 'none'],
        'help':'activation for drug QED prediction output'},
    # Training and validation parameters ######################################
    # Drug response regression training parameters
    {'name': 'resp_loss_func',
        'default': 'mse',
        'choices': ['mse', 'l1'],
        'help':'loss function for drug response regression'},
    {'name': 'resp_opt',
        'default': 'SGD',
        'choices': ['SGD', 'RMSprop', 'Adam'],
        'help':'optimizer for drug response regression'},
    {'name': 'resp_lr',
        'default': 1e-5,
        'type': float,
        'help': 'learning rate for drug response regression'},
    # Cell line classification training parameters
    {'name': 'cl_clf_opt',
        'default': 'SGD',
        'choices': ['SGD', 'RMSprop', 'Adam'],
        'help':'optimizer for cell line classification'},
    {'name': 'cl_clf_lr',
        'default': 1e-3,
        'type': float,
        'help': 'learning rate for cell line classification'},
    # Drug target family classification training parameters
    {'name': 'drug_target_opt',
        'default': 'SGD',
        'choices': ['SGD', 'RMSprop', 'Adam'],
        'help':'optimizer for drug target classification training'},
    {'name': 'drug_target_lr',
        'default': 1e-3,
        'type': float,
        'help': 'learning rate for drug target classification'},
    # Drug weighted QED regression training parameters
    {'name': 'drug_qed_loss_func',
        'default': 'mse',
        'choices': ['mse', 'l1'],
        'help':'loss function for drug QED regression'},
    {'name': 'drug_qed_opt',
        'default': 'SGD',
        'choices': ['SGD', 'RMSprop', 'Adam'],
        'help':'optimizer for drug QED regression'},
    {'name': 'drug_qed_lr',
        'default': 1e-3,
        'type': float,
        'help': 'learning rate for drug drug QED regression'},
    # Starting epoch for drug response validation
    {'name': 'resp_val_start_epoch',
        'default': 0,
        'type': int,
        'help': 'starting epoch for drug response validation'},
    # Early stopping based on R2 score of drug response prediction
    {'name': 'early_stop_patience',
        'default': 5,
        'type': int,
        'help': 'patience for early stopping based on drug response validation R2 scores'},
    # Global/shared training parameters
    {'name': 'l2_regularization',
        'default': 1e-5,
        'type': float,
        'help': 'L2 regularization for nn weights'},
    {'name': 'lr_decay_factor',
        'default': 0.95,
        'type': float,
        'help': 'decay factor for learning rate'},
    {'name': 'trn_batch_size',
        'default': 32,
        'type': int,
        'help': 'input batch size for training'},
    {'name': 'val_batch_size',
        'default': 256,
        'type': int,
        'help': 'input batch size for validation'},
    {'name': 'max_num_batches',
        'default': 1000,
        'type': int,
        'help': 'maximum number of batches per epoch'},
    # Miscellaneous settings ##################################################
    {'name': 'multi_gpu',
        'type': candle.str2bool,
        'default': False,
        'help': 'enables multiple GPU process'},
    {'name': 'no_cuda',
        'type': candle.str2bool,
        'default': False,
        'help': 'disables CUDA training'}
]

required = [
    'train_sources',
    'val_sources',
    'gene_layer_dim',
    'gene_num_layers',
    'gene_latent_dim',
    'drug_layer_dim',
    'drug_num_layers',
    'drug_latent_dim',
    'resp_layer_dim',
    'resp_num_layers_per_block',
    'resp_num_blocks',
    'resp_num_layers',
    'dropout',
    'resp_activation',
    'cl_clf_layer_dim',
    'cl_clf_num_layers',
    'drug_target_layer_dim',
    'drug_target_num_layers',
    'drug_qed_layer_dim',
    'drug_qed_num_layers',
    'drug_qed_activation',
    'resp_loss_func',
    'resp_opt',
    'resp_lr',
    'cl_clf_opt',
    'cl_clf_lr',
    'drug_target_opt',
    'drug_target_lr',
    'drug_qed_loss_func',
    'drug_qed_opt',
    'drug_qed_lr',
    'trn_batch_size',
    'val_batch_size',
    'epochs',
    'rng_seed',
    'val_split',
    'timeout',
]


class unoMTBk(candle.Benchmark):

    def set_locals(self):
        """Functionality to set variables specific for the benchmark
        - required: set of required parameters for the benchmark.
        - additional_definitions: list of dictionaries describing the additional parameters for the
        benchmark.
        """

        if required is not None:
            self.required = set(required)
        if additional_definitions is not None:
            self.additional_definitions = additional_definitions
