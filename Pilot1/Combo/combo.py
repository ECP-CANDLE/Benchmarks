from __future__ import print_function

import os
import sys
import logging

file_path = os.path.dirname(os.path.realpath(__file__))
lib_path2 = os.path.abspath(os.path.join(file_path, '..', '..', 'common'))
sys.path.append(lib_path2)

import candle

logger = logging.getLogger(__name__)

additional_definitions = [
    {'name': 'cell_features',
     'nargs': '+',
     'choices': ['expression', 'mirna', 'proteome', 'all', 'expression_5platform', 'expression_u133p2', 'rnaseq', 'categorical'],
     'help': "use one or more cell line feature sets: 'expression', 'mirna', 'proteome', 'all'; use all for ['expression', 'mirna', 'proteome']; use 'categorical' for one-hot encoded cell lines"},
    {'name': 'drug_features',
     'nargs': '+',
     'choices': ['descriptors', 'latent', 'all', 'categorical', 'noise'],
     'help': "use dragon7 descriptors, latent representations from Aspuru-Guzik's SMILES autoencoder, or both, or one-hot encoded drugs, or random features; 'descriptors','latent', 'all', 'categorical', 'noise'"},
    {'name': 'dense_feature_layers',
     'nargs': '+',
     'type': int,
     'help': 'number of neurons in intermediate dense layers in the feature encoding submodels'},
    {'name': 'use_landmark_genes',
     'type': candle.str2bool,
     'default': True,
     'help': "use the 978 landmark genes from LINCS (L1000) as expression features"},
    {'name': 'use_combo_score',
     'type': candle.str2bool,
     'default': False,
     'help': "use combination score in place of percent growth (stored in 'GROWTH' column)"},
    {'name': 'preprocess_rnaseq',
     'default': 'none',
     'choices': ['source_scale', 'combat', 'none'],
     'help': "preprocessing method for RNAseq data; none for global normalization"},
    {'name': 'response_url',
     'default': None,
     'help': "URL to combo dose response file"},
    {'name': 'residual',
     'type': candle.str2bool,
     'default': True,
     'help': "add skip connections to the layers"},
    {'name': 'reduce_lr',
     'type': candle.str2bool,
     'default': True,
     'help': 'reduce learning rate on plateau'},
    {'name': 'warmup_lr',
     'type': candle.str2bool,
     'default': True,
     'help': 'gradually increase learning rate on start'},
    {'name': 'base_lr',
     'type': float,
     'default': None,
     'help': 'base learning rate'},
    {'name': 'cp',
     'type': candle.str2bool,
     'default': True,
     'help': 'checkpoint models with best val_loss'},
    {'name': 'tb',
     'type': candle.str2bool,
     'default': True,
     'help': 'use tensorboard'},
    {'name': 'use_mean_growth',
     'type': candle.str2bool,
     'default': False,
     'help': 'aggregate growth percentage by mean instead of min'},
    {'name': 'max_val_loss',
     'type': float,
     'help': 'retrain if val_loss is greater than the threshold'},
    {'name': 'cv_partition',
     'choices': ['overlapping', 'disjoint', 'disjoint_cells'],
     'help': "cross validation paritioning scheme: overlapping or disjoint"},
    {'name': 'cv',
     'type': int,
     'help': "cross validation folds"},
    {'name': 'gen',
     'type': candle.str2bool,
     'default': True,
     'help': "use generator for training and validation data"},
    {'name': 'exclude_cells',
     'nargs': '+',
     'default': [],
     'help': "cell line IDs to exclude"},
    {'name': 'exclude_drugs',
     'nargs': '+',
     'default': [],
     'help': "drug line IDs to exclude"}
]

required = ['activation', 'batch_size', 'dense', 'dense_feature_layers', 'dropout',
            'epochs', 'learning_rate', 'loss', 'optimizer', 'residual', 'rng_seed',
            'save_path', 'scaling', 'feature_subsample', 'val_split',
            'timeout']


class BenchmarkCombo(candle.Benchmark):
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
