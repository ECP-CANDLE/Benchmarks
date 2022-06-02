"""
    File Name:          UnoPytorch/encoder_init.py
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               8/16/18
    Python Version:     3.6.6
    File Description:

"""
import logging
import copy
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from networks.structures.encoder_net import EncNet
from utils.data_processing.cell_line_dataframes import get_rna_seq_df
from utils.data_processing.drug_dataframes import get_drug_feature_df

from utils.datasets.basic_dataset import DataFrameDataset
from utils.miscellaneous.optimizer import get_optimizer
from utils.miscellaneous.random_seeding import seed_random_state

logger = logging.getLogger(__name__)


def get_encoder(
        model_path: str,
        dataframe: pd.DataFrame,

        # Autoencoder network configuration
        autoencoder_init: bool,
        layer_dim: int,
        num_layers: int,
        latent_dim: int,

        # Major training parameters
        ae_loss_func: str,
        ae_opt: str,
        ae_lr: float,
        ae_reg: float,
        lr_decay_factor: float,
        max_num_epochs: int,
        early_stop_patience: int,

        # Secondary training parameters
        validation_ratio: float = 0.2,
        trn_batch_size: int = 32,
        val_batch_size: int = 1024,

        # Miscellaneous
        device: torch.device = torch.device('cuda'),
        verbose: bool = True,
        rand_state: int = 0, ):
    """encoder = gene_encoder = get_encoder(./models/', dataframe,
           True, 1000, 3, 100, 'mse', 'sgd', 1e-3, 0.98, 100, 10)

    This function constructs, initializes and returns a feature encoder for
    the given dataframe.

    When parameter autoencoder_init is set to False, it simply construct and
    return an encoder with simple initialization (nn.init.xavier_normal_).

    When autoencoder_init is set to True, the function will return the
    encoder part of an autoencoder trained on the given data. It will first
    check if the model file exists. If not, it will start training with
    given training hyper-parameters.

    Note that the saved model in disk contains the whole autoencoder (
    encoder and decoder). But the function only returns the encoder.

    Also the dataframe should be processed before this function call.

    Args:
        model_path (str): path to model for loading (if exists) and
            saving (for future usage).
        dataframe (pd.DataFrame): dataframe for training and validation.

        autoencoder_init (bool): indicator for using autoencoder as feature
            encoder initialization method. If True, the function will
            construct a autoencoder with symmetric encoder and decoder,
            and then train it with part of dataframe while validating on
            the rest, until early stopping is evoked or running out of epochs.
        layer_dim (int): layer dimension for feature encoder.
        num_layers (int): number of layers for feature encoder.
        latent_dim (int): latent (output) space dimension for feature encoder.

        ae_loss_func (str): loss function for autoencoder training. Select
            between 'mse' and 'l1'.
        ae_opt (str): optimizer for autoencoder training. Select between
            'SGD', 'Adam', and 'RMSprop'.
        ae_lr (float): learning rate for autoencoder training.
        lr_decay_factor (float): exponential learning rate decay factor.
        max_num_epochs (int): maximum number of epochs allowed.
        early_stop_patience (int): patience for early stopping. If the
            validation loss does not increase for this many epochs, the
            function returns the encoder part of the autoencoder, with the
            best validation loss so far.

        validation_ratio (float): (validation data size / overall data size).
        trn_batch_size (int): batch size for training.
        val_batch_size (int): batch size for validation.

        device (torch.device): torch device indicating where to train:
            either on CPU or GPU. Note that this function does not support
            multi-GPU yet.
        verbose (bool): indicator for training epoch log on terminal.
        rand_state (int): random seed used for layer initialization,
            training/validation splitting, and all other processes that
            requires randomness.

    Returns:
        torch.nn.Module: encoder for features from given dataframe.
    """

    # If autoencoder initialization is not required, return a plain encoder
    if not autoencoder_init:
        return EncNet(input_dim=dataframe.shape[1],
                      layer_dim=layer_dim,
                      num_layers=num_layers,
                      latent_dim=latent_dim,
                      autoencoder=False).to(device).encoder

    # Check if the model exists, load and return
    if os.path.exists(model_path):
        logger.debug('Loading existing autoencoder model from %s ...'
                     % model_path)

        model = EncNet(input_dim=dataframe.shape[1],
                       layer_dim=layer_dim,
                       latent_dim=latent_dim,
                       num_layers=num_layers,
                       autoencoder=True).to(device)
        model.load_state_dict(torch.load(model_path))
        return model.encoder

    logger.debug('Constructing autoencoder from dataframe ...')

    # Setting up random seed for reproducible and deterministic results
    seed_random_state(rand_state)

    # Load dataframe, split and construct dataloaders #########################
    trn_df, val_df = train_test_split(dataframe,
                                      test_size=validation_ratio,
                                      random_state=rand_state,
                                      shuffle=True)
    dataloader_kwargs = {
        'shuffle': 'True',
        'num_workers': 4 if device == torch.device('cuda') else 0,
        'pin_memory': True if device == torch.device('cuda') else False, }

    trn_dataloader = DataLoader(DataFrameDataset(trn_df),
                                batch_size=trn_batch_size,
                                **dataloader_kwargs)

    val_dataloader = DataLoader(DataFrameDataset(val_df),
                                batch_size=val_batch_size,
                                **dataloader_kwargs)

    # Construct the network and get prepared for training #####################
    autoencoder = EncNet(input_dim=dataframe.shape[1],
                         layer_dim=layer_dim,
                         latent_dim=latent_dim,
                         num_layers=num_layers,
                         autoencoder=True).to(device)
    assert ae_loss_func.lower() == 'l1' or ae_loss_func.lower() == 'mse'
    loss_func = F.l1_loss if ae_loss_func.lower() == 'l1' else F.mse_loss

    optimizer = get_optimizer(opt_type=ae_opt,
                              networks=autoencoder,
                              learning_rate=ae_lr,
                              l2_regularization=ae_reg)
    lr_decay = LambdaLR(optimizer, lr_lambda=lambda e: lr_decay_factor ** e)

    # Train until max number of epochs is reached or early stopped ############
    best_val_loss = np.inf
    best_autoencoder = None
    patience = 0

    if verbose:
        print('=' * 80)
        print('Training log for autoencoder model (%s): ' % model_path)

    for epoch in range(max_num_epochs):

        lr_decay.step(epoch)

        # Training loop for autoencoder
        autoencoder.train()
        trn_loss = 0.
        for batch_idx, samples in enumerate(trn_dataloader):
            samples = samples.to(device)
            recon_samples = autoencoder(samples)
            autoencoder.zero_grad()

            loss = loss_func(input=recon_samples, target=samples)
            loss.backward()
            optimizer.step()

            trn_loss += loss.item() * len(samples)
        trn_loss /= len(trn_dataloader.dataset)

        # Validation loop for autoencoder
        autoencoder.eval()
        val_loss = 0.
        with torch.no_grad():
            for samples in val_dataloader:
                samples = samples.to(device)
                recon_samples = autoencoder(samples)
                loss = loss_func(input=recon_samples, target=samples)

                val_loss += loss.item() * len(samples)
            val_loss /= len(val_dataloader.dataset)

        if verbose:
            print('Epoch %4i: training loss: %.4f;\t validation loss: %.4f'
                  % (epoch + 1, trn_loss, val_loss))

        # Save the model to memory if it achieves best validation loss
        if val_loss < best_val_loss:
            patience = 0
            best_val_loss = val_loss
            best_autoencoder = copy.deepcopy(autoencoder)
        # Otherwise increase patience and check for early stopping
        else:
            patience += 1
            if patience > early_stop_patience:
                if verbose:
                    print('Evoking early stopping. Best validation loss %.4f.'
                          % best_val_loss)
                break

    # Store the best autoencoder and return it ################################
    try:
        os.makedirs(os.path.dirname(model_path))
    except FileExistsError:
        pass
    torch.save(best_autoencoder.state_dict(), model_path)
    return best_autoencoder.encoder


def get_gene_encoder(
        model_folder: str,
        data_root: str,

        # RNA sequence usage and scaling
        rnaseq_feature_usage: str,
        rnaseq_scaling: str,

        # Autoencoder network configuration
        autoencoder_init: bool,
        layer_dim: int,
        num_layers: int,
        latent_dim: int,

        # Training keyword parameters to be provided
        training_kwarg: dict,

        # Miscellaneous
        device: torch.device = torch.device('cuda'),
        verbose: bool = True,
        rand_state: int = 0, ):
    """gene_encoder = get_gene_encoder(./models/', './data/',
           'source_scale', 'std', True, 1000, 3, 100, training_kwarg_dict)

    This function takes arguments about RNA sequence encoder and return the
    corresponding encoders. It will execute one of the following based on
    parameters and previous saved models:
        * simply initialize a new encoder;
        * load existing autoencoder and return the encoder part;
        * train a new autoencoder and return the encoder part;

    Note that this function requires existing dataframes of RNA sequence.

    Args:
        model_folder (str): path to the model folder.
        data_root (str): path to data folder (root).

        rnaseq_feature_usage (str): RNA sequence data used. Choose between
            'source_scale' and 'combat'.
        rnaseq_scaling (str): Scaling method for RNA sequence data.

        autoencoder_init (bool): indicator for using autoencoder as RNA
            sequence encoder initialization method.
        layer_dim (int): layer dimension for RNA sequence encoder.
        num_layers (int): number of layers for RNA sequence encoder.
        latent_dim (int): latent (output) space dimension for RNA sequence
            encoder.

        training_kwarg (dict): training parameters in dict format,
            which contains all the training parameters in get_encoder
            function. Please refer to get_encoder for more details.

        device (torch.device): torch device indicating where to train:
            either on CPU or GPU. Note that this function does not support
            multi-GPU yet.
        verbose (bool): indicator for training epoch log on terminal.
        rand_state (int): random seed used for layer initialization,
            training/validation splitting, and all other processes that
            requires randomness.

    Returns:
        torch.nn.Module: encoder for RNA sequence dataframe.
    """

    gene_encoder_name = 'gene_net(%i*%i=>%i, %s, scaling=%s).pt' % \
                        (layer_dim, num_layers, latent_dim,
                         rnaseq_feature_usage, rnaseq_scaling)
    gene_encoder_path = os.path.join(model_folder, gene_encoder_name)

    rna_seq_df = get_rna_seq_df(data_root=data_root,
                                rnaseq_feature_usage=rnaseq_feature_usage,
                                rnaseq_scaling=rnaseq_scaling)
    rna_seq_df.drop_duplicates(inplace=True)

    return get_encoder(
        model_path=gene_encoder_path,
        dataframe=rna_seq_df,

        autoencoder_init=autoencoder_init,
        layer_dim=layer_dim,
        num_layers=num_layers,
        latent_dim=latent_dim,

        **training_kwarg,

        device=device,
        verbose=verbose,
        rand_state=rand_state, )


def get_drug_encoder(
        model_folder: str,
        data_root: str,

        # Drug feature usage and scaling
        drug_feature_usage: str,
        dscptr_scaling: str,
        dscptr_nan_threshold: float,

        # Autoencoder network configuration
        autoencoder_init: bool,
        layer_dim: int,
        num_layers: int,
        latent_dim: int,

        # Training keyword parameters to be provided
        training_kwarg: dict,

        # Miscellaneous
        device: torch.device = torch.device('cuda'),
        verbose: bool = True,
        rand_state: int = 0, ):
    """drug_encoder = get_gene_encoder(./models/', './data/',
               'both', 'std', 0.,  True, 1000, 3, 100, training_kwarg_dict)

    This function takes arguments about drug feature encoder and return the
    corresponding encoders. It will execute one of the following based on
    parameters and previous saved models:
        * simply initialize a new encoder;
        * load existing autoencoder and return the encoder part;
        * train a new autoencoder and return the encoder part;

    Note that this function requires existing dataframes of drug feature.


    Args:
        model_folder (str): path to the model folder.
        data_root (str): path to data folder (root).

        drug_feature_usage (str): Drug feature usage used. Choose between
            'fingerprint', 'descriptor', or 'both'.
        dscptr_scaling (str): Scaling method for drug feature data.
        dscptr_nan_threshold (float): ratio of NaN values allowed for drug
            features. Unqualified columns and rows will be dropped.

        autoencoder_init (bool): indicator for using autoencoder as drug
            feature encoder initialization method.
        layer_dim (int): layer dimension for drug feature encoder.
        num_layers (int): number of layers for drug feature encoder.
        latent_dim (int): latent (output) space dimension for drug feature
            encoder.

        training_kwarg (dict): training parameters in dict format,
            which contains all the training parameters in get_encoder
            function. Please refer to get_encoder for more details.

        device (torch.device): torch device indicating where to train:
            either on CPU or GPU. Note that this function does not support
            multi-GPU yet.
        verbose (bool): indicator for training epoch log on terminal.
        rand_state (int): random seed used for layer initialization,
            training/validation splitting, and all other processes that
            requires randomness.

    Returns:
        torch.nn.Module: encoder for drug feature dataframe.
    """

    drug_encoder_name = 'drug_net(%i*%i=>%i, %s, descriptor_scaling=%s, ' \
                        'nan_thresh=%.2f).pt' % \
                        (layer_dim, num_layers, latent_dim,
                         drug_feature_usage, dscptr_scaling,
                         dscptr_nan_threshold,)
    drug_encoder_path = os.path.join(model_folder, drug_encoder_name)

    drug_feature_df = get_drug_feature_df(
        data_root=data_root,
        drug_feature_usage=drug_feature_usage,
        dscptr_scaling=dscptr_scaling,
        dscptr_nan_thresh=dscptr_nan_threshold)

    return get_encoder(
        model_path=drug_encoder_path,
        dataframe=drug_feature_df,

        autoencoder_init=autoencoder_init,
        layer_dim=layer_dim,
        num_layers=num_layers,
        latent_dim=latent_dim,

        **training_kwarg,

        device=device,
        verbose=verbose,
        rand_state=rand_state, )


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    # Test code for autoencoder with RNA sequence and drug features
    ae_training_kwarg = {
        'ae_loss_func': 'mse',
        'ae_opt': 'sgd',
        'ae_lr': 0.2,
        'lr_decay_factor': 1.0,
        'max_num_epochs': 1000,
        'early_stop_patience': 50, }

    gene_encoder = get_gene_encoder(
        model_folder='../../models/',
        data_root='../../data/',

        rnaseq_feature_usage='source_scale',
        rnaseq_scaling='std',

        autoencoder_init=True,
        layer_dim=1024,
        num_layers=2,
        latent_dim=512,

        training_kwarg=ae_training_kwarg,

        device=torch.device('cuda'),
        verbose=True,
        rand_state=0, )

    drug_encoder = get_drug_encoder(
        model_folder='../../models/',
        data_root='../../data/',

        drug_feature_usage='both',
        dscptr_scaling='std',
        dscptr_nan_threshold=0.0,

        autoencoder_init=True,
        layer_dim=4096,
        num_layers=2,
        latent_dim=2048,

        training_kwarg=ae_training_kwarg,

        device=torch.device('cuda'),
        verbose=True,
        rand_state=0, )
