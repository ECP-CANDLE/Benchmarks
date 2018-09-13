""" 
    File Name:          UnoPytorch/uno_pytorch.py
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               8/13/18
    Python Version:     3.6.6
    File Description:   

"""

import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.core.multiarray import ndarray
from torch.optim.lr_scheduler import LambdaLR

from networks.functions.cl_clf_func import train_cl_clf, valid_cl_clf
from networks.functions.drug_qed_func import train_drug_qed, valid_drug_qed
from networks.functions.drug_target_func import train_drug_target, \
    valid_drug_target
from networks.functions.resp_func import train_resp, valid_resp
from networks.structures.classification_net import ClfNet
from networks.structures.regression_net import RgsNet
from networks.structures.response_net import RespNet
from utils.data_processing.label_encoding import get_label_dict
from utils.datasets.drug_qed_dataset import DrugQEDDataset
from utils.datasets.drug_resp_dataset import DrugRespDataset
from utils.datasets.cl_class_dataset import CLClassDataset
from utils.data_processing.dataframe_scaling import SCALING_METHODS
from networks.initialization.encoder_init import get_gene_encoder, \
    get_drug_encoder
from utils.datasets.drug_target_dataset import DrugTargetDataset
from utils.miscellaneous.optimizer import get_optimizer
from utils.miscellaneous.random_seeding import seed_random_state


# Number of workers for dataloader. Too many workers might lead to process
# hanging for PyTorch version 4.1. Set this number between 0 and 4.
NUM_WORKER = 4
DATA_ROOT = './data/'


def main():
    # Training settings and hyper-parameters
    parser = argparse.ArgumentParser(
        description='Multitasking Neural Network for Genes and Drugs')

    # Dataset parameters ######################################################
    # Training and validation data sources
    parser.add_argument('--trn_src', type=str, required=True,
                        help='training source for drug response')
    parser.add_argument('--val_srcs', type=str, required=True, nargs='+',
                        help='validation list of sources for drug response')

    # Pre-processing for dataframes
    parser.add_argument('--grth_scaling', type=str, default='std',
                        help='scaling method for drug response (growth)',
                        choices=SCALING_METHODS)
    parser.add_argument('--dscptr_scaling', type=str, default='std',
                        help='scaling method for drug feature (descriptor)',
                        choices=SCALING_METHODS)
    parser.add_argument('--rnaseq_scaling', type=str, default='std',
                        help='scaling method for RNA sequence',
                        choices=SCALING_METHODS)
    parser.add_argument('--dscptr_nan_threshold', type=float, default=0.0,
                        help='ratio of NaN values allowed for drug descriptor')
    parser.add_argument('--qed_scaling', type=str, default='none',
                        help='scaling method for drug weighted QED',
                        choices=SCALING_METHODS)

    # Feature usage and partitioning settings
    parser.add_argument('--rnaseq_feature_usage', type=str, default='combat',
                        help='RNA sequence data used',
                        choices=['source_scale', 'combat', ])
    parser.add_argument('--drug_feature_usage', type=str, default='both',
                        help='drug features (fp and/or desc) used',
                        choices=['fingerprint', 'descriptor', 'both', ])
    parser.add_argument('--validation_ratio', type=float, default=0.2,
                        help='ratio for validation dataset')
    parser.add_argument('--disjoint_drugs', action='store_true',
                        help='disjoint drugs between train/validation')
    parser.add_argument('--disjoint_cells', action='store_true',
                        help='disjoint cells between train/validation')

    # Network configuration ###################################################
    # Encoders for drug features and RNA sequence (LINCS 1000)
    parser.add_argument('--gene_layer_dim', type=int, default=1024,
                        help='dimension of layers for RNA sequence')
    parser.add_argument('--gene_latent_dim', type=int, default=256,
                        help='dimension of latent variable for RNA sequence')
    parser.add_argument('--gene_num_layers', type=int, default=2,
                        help='number of layers for RNA sequence')

    parser.add_argument('--drug_layer_dim', type=int, default=4096,
                        help='dimension of layers for drug feature')
    parser.add_argument('--drug_latent_dim', type=int, default=1024,
                        help='dimension of latent variable for drug feature')
    parser.add_argument('--drug_num_layers', type=int, default=2,
                        help='number of layers for drug feature')

    # Using autoencoder for drug/sequence encoder initialization
    parser.add_argument('--autoencoder_init', action='store_true',
                        help='indicator of autoencoder initialization for '
                             'drug/RNA sequence feature encoder')

    # Drug response regression network
    parser.add_argument('--resp_layer_dim', type=int, default=1024,
                        help='dimension of layers for drug response block')
    parser.add_argument('--resp_num_layers_per_block', type=int, default=2,
                        help='number of layers for drug response res block')
    parser.add_argument('--resp_num_blocks', type=int, default=2,
                        help='number of residual blocks for drug response')
    parser.add_argument('--resp_num_layers', type=int, default=2,
                        help='number of layers for drug response')
    parser.add_argument('--resp_dropout', type=float, default=0.0,
                        help='dropout of residual blocks for drug response')
    parser.add_argument('--resp_activation', type=str, default='none',
                        help='activation for response prediction output',
                        choices=['sigmoid', 'tanh', 'none'])

    # Cell line classification network(s)
    parser.add_argument('--cl_clf_layer_dim', type=int, default=256,
                        help='layer dimension for cell line classification')
    parser.add_argument('--cl_clf_num_layers', type=int, default=1,
                        help='number of layers for cell line classification')

    # Drug target family classification network
    parser.add_argument('--drug_target_layer_dim', type=int, default=512,
                        help='dimension of layers for drug target prediction')
    parser.add_argument('--drug_target_num_layers', type=int, default=2,
                        help='number of layers for drug target prediction')

    # Drug weighted QED regression network
    parser.add_argument('--drug_qed_layer_dim', type=int, default=512,
                        help='dimension of layers for drug QED prediction')
    parser.add_argument('--drug_qed_num_layers', type=int, default=2,
                        help='number of layers for drug QED prediction')
    parser.add_argument('--drug_qed_activation', type=str, default='none',
                        help='activation for drug QED prediction output',
                        choices=['sigmoid', 'tanh', 'none'])

    # Training and validation parameters ######################################
    # Drug response regression training parameters
    parser.add_argument('--resp_loss_func', type=str, default='mse',
                        help='loss function for drug response regression',
                        choices=['mse', 'l1'])
    parser.add_argument('--resp_opt', type=str, default='SGD',
                        help='optimizer for drug response regression',
                        choices=['SGD', 'RMSprop', 'Adam'])
    parser.add_argument('--resp_lr', type=float, default=1e-5,
                        help='learning rate for drug response regression')

    # Cell line classification training parameters
    parser.add_argument('--cl_clf_opt', type=str, default='SGD',
                        help='optimizer for cell line classification',
                        choices=['SGD', 'RMSprop', 'Adam'])
    parser.add_argument('--cl_clf_lr', type=float, default=1e-3,
                        help='learning rate for cell line classification')

    # Drug target family classification training parameters
    parser.add_argument('--drug_target_opt', type=str, default='SGD',
                        help='optimizer for drug target classification '
                             'training',
                        choices=['SGD', 'RMSprop', 'Adam'])
    parser.add_argument('--drug_target_lr', type=float, default=1e-3,
                        help='learning rate for drug target classification')

    # Drug weighted QED regression training parameters
    parser.add_argument('--drug_qed_loss_func', type=str, default='mse',
                        help='loss function for drug QED regression',
                        choices=['mse', 'l1'])
    parser.add_argument('--drug_qed_opt', type=str, default='SGD',
                        help='optimizer for drug rQED regression',
                        choices=['SGD', 'RMSprop', 'Adam'])
    parser.add_argument('--drug_qed_lr', type=float, default=1e-3,
                        help='learning rate for drug QED regression')

    # Starting epoch for drug response validation
    parser.add_argument('--resp_val_start_epoch', type=int, default=0,
                        help='starting epoch for drug response validation')

    # Early stopping based on R2 score of drug response prediction
    parser.add_argument('--early_stop_patience', type=int, default=5,
                        help='patience for early stopping based on drug '
                             'response validation R2 scores ')

    # Global/shared training parameters
    parser.add_argument('--l2_regularization', type=float, default=1e-5,
                        help='L2 regularization for nn weights')
    parser.add_argument('--lr_decay_factor', type=float, default=0.95,
                        help='decay factor for learning rate')
    parser.add_argument('--trn_batch_size', type=int, default=32,
                        help='input batch size for training')
    parser.add_argument('--val_batch_size', type=int, default=256,
                        help='input batch size for validation')
    parser.add_argument('--max_num_batches', type=int, default=1000,
                        help='maximum number of batches per epoch')
    parser.add_argument('--max_num_epochs', type=int, default=100,
                        help='maximum number of epochs')

    # Miscellaneous settings ##################################################
    parser.add_argument('--multi_gpu', action='store_true', default=False,
                        help='enables multiple GPU process')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--rand_state', type=int, default=0,
                        help='random state of numpy/sklearn/pytorch')

    args = parser.parse_args()
    print('Training Arguments:\n' + json.dumps(vars(args), indent=4))

    # Setting up random seed for reproducible and deterministic results
    seed_random_state(args.rand_state)

    # Computation device config (cuda or cpu)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    # Data loaders for training/validation ####################################
    dataloader_kwargs = {
        'timeout': 1,
        'shuffle': 'True',
        # 'num_workers': multiprocessing.cpu_count() if use_cuda else 0,
        'num_workers': NUM_WORKER if use_cuda else 0,
        'pin_memory': True if use_cuda else False, }

    # Drug response dataloaders for training/validation
    drug_resp_dataset_kwargs = {
        'data_root': DATA_ROOT,
        'rand_state': args.rand_state,
        'summary': False,

        'int_dtype': np.int8,
        'float_dtype': np.float16,
        'output_dtype': np.float32,

        'grth_scaling': args.grth_scaling,
        'dscptr_scaling': args.dscptr_scaling,
        'rnaseq_scaling': args.rnaseq_scaling,
        'dscptr_nan_threshold': args.dscptr_nan_threshold,

        'rnaseq_feature_usage': args.rnaseq_feature_usage,
        'drug_feature_usage': args.drug_feature_usage,
        'validation_ratio': args.validation_ratio,
        'disjoint_drugs': args.disjoint_drugs,
        'disjoint_cells': args.disjoint_cells, }

    drug_resp_trn_loader = torch.utils.data.DataLoader(
        DrugRespDataset(data_src=args.trn_src,
                        training=True,
                        **drug_resp_dataset_kwargs),
        batch_size=args.trn_batch_size,
        **dataloader_kwargs)

    # List of data loaders for different validation sets
    drug_resp_val_loaders = [torch.utils.data.DataLoader(
        DrugRespDataset(data_src=src,
                        training=False,
                        **drug_resp_dataset_kwargs),
        batch_size=args.val_batch_size,
        **dataloader_kwargs) for src in args.val_srcs]

    # Cell line classification dataloaders for training/validation
    cl_clf_dataset_kwargs = {
        'data_root': DATA_ROOT,
        'rand_state': args.rand_state,
        'summary': False,

        'int_dtype': np.int8,
        'float_dtype': np.float16,
        'output_dtype': np.float32,

        'rnaseq_scaling': args.rnaseq_scaling,

        'rnaseq_feature_usage': args.rnaseq_feature_usage,
        'validation_ratio': args.validation_ratio, }

    cl_clf_trn_loader = torch.utils.data.DataLoader(
        CLClassDataset(training=True,
                       **cl_clf_dataset_kwargs),
        batch_size=args.trn_batch_size,
        **dataloader_kwargs)

    cl_clf_val_loader = torch.utils.data.DataLoader(
        CLClassDataset(training=False,
                       **cl_clf_dataset_kwargs),
        batch_size=args.val_batch_size,
        **dataloader_kwargs)

    # Drug target family classification dataloaders for training/validation
    drug_target_dataset_kwargs = {
        'data_root': DATA_ROOT,
        'rand_state': args.rand_state,
        'summary': False,

        'int_dtype': np.int8,
        'float_dtype': np.float16,
        'output_dtype': np.float32,

        'dscptr_scaling': args.dscptr_scaling,
        'dscptr_nan_threshold': args.dscptr_nan_threshold,

        'drug_feature_usage': args.drug_feature_usage,
        'validation_ratio': args.validation_ratio, }

    drug_target_trn_loader = torch.utils.data.DataLoader(
        DrugTargetDataset(training=True,
                          **drug_target_dataset_kwargs),
        batch_size=args.trn_batch_size,
        **dataloader_kwargs)

    drug_target_val_loader = torch.utils.data.DataLoader(
        DrugTargetDataset(training=False,
                          **drug_target_dataset_kwargs),
        batch_size=args.val_batch_size,
        **dataloader_kwargs)

    # Drug weighted QED regression dataloaders for training/validation
    drug_qed_dataset_kwargs = {
        'data_root': DATA_ROOT,
        'rand_state': args.rand_state,
        'summary': False,

        'int_dtype': np.int8,
        'float_dtype': np.float16,
        'output_dtype': np.float32,

        'qed_scaling': args.qed_scaling,
        'dscptr_scaling': args.dscptr_scaling,
        'dscptr_nan_threshold': args.dscptr_nan_threshold,

        'drug_feature_usage': args.drug_feature_usage,
        'validation_ratio': args.validation_ratio, }

    drug_qed_trn_loader = torch.utils.data.DataLoader(
        DrugQEDDataset(training=True,
                       **drug_qed_dataset_kwargs),
        batch_size=args.trn_batch_size,
        **dataloader_kwargs)

    drug_qed_val_loader = torch.utils.data.DataLoader(
        DrugQEDDataset(training=False,
                       **drug_qed_dataset_kwargs),
        batch_size=args.val_batch_size,
        **dataloader_kwargs)

    # Constructing and initializing neural networks ###########################
    # Autoencoder training hyper-parameters
    ae_training_kwarg = {
        'ae_loss_func': 'mse',
        'ae_opt': 'sgd',
        'ae_lr': 2e-1,
        'lr_decay_factor': 1.0,
        'max_num_epochs': 1000,
        'early_stop_patience': 50, }

    encoder_kwarg = {
        'model_folder': './models/',
        'data_root': DATA_ROOT,

        'autoencoder_init': args.autoencoder_init,
        'training_kwarg': ae_training_kwarg,

        'device': device,
        'verbose': True,
        'rand_state': args.rand_state, }

    # Get RNA sequence encoder
    gene_encoder = get_gene_encoder(
        rnaseq_feature_usage=args.rnaseq_feature_usage,
        rnaseq_scaling=args.rnaseq_scaling,

        layer_dim=args.gene_layer_dim,
        num_layers=args.gene_num_layers,
        latent_dim=args.gene_latent_dim,
        **encoder_kwarg)

    # Get drug feature encoder
    drug_encoder = get_drug_encoder(
        drug_feature_usage=args.drug_feature_usage,
        dscptr_scaling=args.dscptr_scaling,
        dscptr_nan_threshold=args.dscptr_nan_threshold,

        layer_dim=args.drug_layer_dim,
        num_layers=args.drug_num_layers,
        latent_dim=args.drug_latent_dim,
        **encoder_kwarg)

    # Regressor for drug response
    resp_net = RespNet(
        gene_latent_dim=args.gene_latent_dim,
        drug_latent_dim=args.drug_latent_dim,

        gene_encoder=gene_encoder,
        drug_encoder=drug_encoder,

        resp_layer_dim=args.resp_layer_dim,
        resp_num_layers_per_block=args.resp_num_layers_per_block,
        resp_num_blocks=args.resp_num_blocks,
        resp_num_layers=args.resp_num_layers,
        resp_dropout=args.resp_dropout,

        resp_activation=args.resp_activation).to(device)

    print(resp_net)

    # Sequence classifier for category, site, and type
    cl_clf_net_kwargs = {
        'encoder': gene_encoder,
        'input_dim': args.gene_latent_dim,
        'condition_dim': len(get_label_dict(DATA_ROOT, 'data_src_dict.txt')),
        'layer_dim': args.cl_clf_layer_dim,
        'num_layers': args.cl_clf_num_layers, }

    category_clf_net = ClfNet(
        num_classes=len(get_label_dict(DATA_ROOT, 'category_dict.txt')),
        **cl_clf_net_kwargs).to(device)
    site_clf_net = ClfNet(
        num_classes=len(get_label_dict(DATA_ROOT, 'site_dict.txt')),
        **cl_clf_net_kwargs).to(device)
    type_clf_net = ClfNet(
        num_classes=len(get_label_dict(DATA_ROOT, 'type_dict.txt')),
        **cl_clf_net_kwargs).to(device)

    # Classifier for drug target family prediction
    drug_target_net = ClfNet(
        encoder=drug_encoder,
        input_dim=args.drug_latent_dim,
        condition_dim=0,
        layer_dim=args.drug_target_layer_dim,
        num_layers=args.drug_target_num_layers,
        num_classes=len(get_label_dict(DATA_ROOT, 'drug_target_dict.txt'))).\
        to(device)

    # Regressor for drug weighted QED prediction
    drug_qed_net = RgsNet(
        encoder=drug_encoder,
        input_dim=args.drug_latent_dim,
        condition_dim=0,
        layer_dim=args.drug_qed_layer_dim,
        num_layers=args.drug_qed_num_layers,
        activation=args.drug_qed_activation).to(device)

    # Multi-GPU settings
    if args.multi_gpu:
        resp_net = nn.DataParallel(resp_net)
        category_clf_net = nn.DataParallel(category_clf_net)
        site_clf_net = nn.DataParallel(site_clf_net)
        type_clf_net = nn.DataParallel(type_clf_net)
        drug_target_net = nn.DataParallel(drug_target_net)
        drug_qed_net = nn.DataParallel(drug_qed_net)

    # Optimizers, learning rate decay, and miscellaneous ######################
    resp_opt = get_optimizer(opt_type=args.resp_opt,
                             networks=resp_net,
                             learning_rate=args.resp_lr,
                             l2_regularization=args.l2_regularization)
    cl_clf_opt = get_optimizer(opt_type=args.cl_clf_opt,
                               networks=[category_clf_net,
                                         site_clf_net,
                                         type_clf_net],
                               learning_rate=args.cl_clf_lr,
                               l2_regularization=args.l2_regularization)
    drug_target_opt = get_optimizer(opt_type=args.drug_target_opt,
                                    networks=drug_target_net,
                                    learning_rate=args.drug_target_lr,
                                    l2_regularization=args.l2_regularization)
    drug_qed_opt = get_optimizer(opt_type=args.drug_qed_opt,
                                 networks=drug_qed_net,
                                 learning_rate=args.drug_qed_lr,
                                 l2_regularization=args.l2_regularization)

    resp_lr_decay = LambdaLR(optimizer=resp_opt,
                             lr_lambda=lambda e:
                             args.lr_decay_factor ** e)
    cl_clf_lr_decay = LambdaLR(optimizer=cl_clf_opt,
                               lr_lambda=lambda e:
                               args.lr_decay_factor ** e)
    drug_target_lr_decay = LambdaLR(optimizer=drug_target_opt,
                                    lr_lambda=lambda e:
                                    args.lr_decay_factor ** e)
    drug_qed_lr_decay = LambdaLR(optimizer=drug_qed_opt,
                                 lr_lambda=lambda e:
                                 args.lr_decay_factor ** e)

    resp_loss_func = F.l1_loss if args.resp_loss_func == 'l1' \
        else F.mse_loss
    drug_qed_loss_func = F.l1_loss if args.drug_qed_loss_func == 'l1' \
        else F.mse_loss

    # Training/validation loops ###############################################
    val_cl_clf_acc = []
    val_drug_target_acc = []
    val_drug_qed_mse, val_drug_qed_mae, val_drug_qed_r2 = [], [], []
    val_resp_mse, val_resp_mae, val_resp_r2 = [], [], []
    best_r2 = -np.inf
    patience = 0
    start_time = time.time()

    # Early stopping is decided on the validation set with the same
    # data source as the training dataloader
    val_index = 0
    for idx, loader in enumerate(drug_resp_val_loaders):
        if loader.dataset.data_source == args.trn_src:
            val_index = idx

    for epoch in range(args.max_num_epochs):

        print('=' * 80 + '\nTraining Epoch %3i:' % (epoch + 1))
        epoch_start_time = time.time()
        resp_lr_decay.step(epoch)
        cl_clf_lr_decay.step(epoch)
        drug_target_lr_decay.step(epoch)
        drug_qed_lr_decay.step(epoch)

        # Training cell line classifier
        train_cl_clf(device=device,
                     category_clf_net=category_clf_net,
                     site_clf_net=site_clf_net,
                     type_clf_net=type_clf_net,
                     data_loader=cl_clf_trn_loader,
                     max_num_batches=args.max_num_batches,
                     optimizer=cl_clf_opt)

        # Training drug target classifier
        train_drug_target(device=device,
                          drug_target_net=drug_target_net,
                          data_loader=drug_target_trn_loader,
                          max_num_batches=args.max_num_batches,
                          optimizer=drug_target_opt)

        # Training drug weighted QED regressor
        train_drug_qed(device=device,
                       drug_qed_net=drug_qed_net,
                       data_loader=drug_qed_trn_loader,
                       max_num_batches=args.max_num_batches,
                       loss_func=drug_qed_loss_func,
                       optimizer=drug_qed_opt)

        # Training drug response regressor
        train_resp(device=device,
                   resp_net=resp_net,
                   data_loader=drug_resp_trn_loader,
                   max_num_batches=args.max_num_batches,
                   loss_func=resp_loss_func,
                   optimizer=resp_opt)

        print('\nValidation Results:')

        if epoch >= args.resp_val_start_epoch:

            # Validating cell line classifier
            cl_category_acc, cl_site_acc, cl_type_acc = \
                valid_cl_clf(device=device,
                             category_clf_net=category_clf_net,
                             site_clf_net=site_clf_net,
                             type_clf_net=type_clf_net,
                             data_loader=cl_clf_val_loader, )
            val_cl_clf_acc.append([cl_category_acc, cl_site_acc, cl_type_acc])

            # Validating drug target classifier
            drug_target_acc = \
                valid_drug_target(device=device,
                                  drug_target_net=drug_target_net,
                                  data_loader=drug_target_val_loader)
            val_drug_target_acc.append(drug_target_acc)

            # Validating drug weighted QED regressor
            drug_qed_mse, drug_qed_mae, drug_qed_r2 = \
                valid_drug_qed(device=device,
                               drug_qed_net=drug_qed_net,
                               data_loader=drug_qed_val_loader)
            val_drug_qed_mse.append(drug_qed_mse)
            val_drug_qed_mae.append(drug_qed_mae)
            val_drug_qed_r2.append(drug_qed_r2)

            # Validating drug response regressor
            resp_mse, resp_mae, resp_r2 = \
                valid_resp(device=device,
                           resp_net=resp_net,
                           data_loaders=drug_resp_val_loaders)

            # Save the validation results in nested list
            val_resp_mse.append(resp_mse)
            val_resp_mae.append(resp_mae)
            val_resp_r2.append(resp_r2)

            # Record the best R2 score (same data source)
            # and check for early stopping if no improvement for epochs
            if resp_r2[val_index] > best_r2:
                patience = 0
                best_r2 = resp_r2[val_index]
            else:
                patience += 1
            if patience >= args.early_stop_patience:
                print('Validation results does not improve for %d epochs ... '
                      'invoking early stopping.' % patience)
                break

        print('Epoch Running Time: %.1f Seconds.'
              % (time.time() - epoch_start_time))

    val_cl_clf_acc = np.array(val_cl_clf_acc).reshape(-1, 3)
    # val_drug_target_acc = np.array(val_drug_target_acc)
    # val_drug_qed_mse = np.array(val_drug_qed_mse)
    # val_resp_mae = np.array(val_resp_mae)
    # val_resp_r2 = np.array(val_resp_r2)
    val_resp_mse, val_resp_mae, val_resp_r2 = \
        np.array(val_resp_mse).reshape(-1, len(args.val_srcs)), \
        np.array(val_resp_mae).reshape(-1, len(args.val_srcs)), \
        np.array(val_resp_r2).reshape(-1, len(args.val_srcs))

    print('Program Running Time: %.1f Seconds.' % (time.time() - start_time))

    # Print overall validation results
    print('=' * 80)
    print('Overall Validation Results:\n')

    print('\tBest Results from Different Models (Epochs):')
    # Print best accuracy for cell line classifiers
    clf_targets = ['Cell Line Categories',
                   'Cell Line Sites',
                   'Cell Line Types', ]
    best_acc = np.amax(val_cl_clf_acc, axis=0)
    best_acc_epochs = np.argmax(val_cl_clf_acc, axis=0)

    for index, clf_target in enumerate(clf_targets):
        print('\t\t%-24s Best Accuracy: %.3f%% (Epoch = %3d)'
              % (clf_target, best_acc[index], best_acc_epochs[index] + 1))

    # Print best predictions for drug classifiers and regressor
    print('\t\tDrug Target Family \t Best Accuracy: %.3f%% (Epoch = %3d)'
          % (np.max(val_drug_target_acc),
             (np.argmax(val_drug_target_acc) + 1)))

    print('\t\tDrug Weighted QED \t Best R2 Score: %+6.4f '
          '(Epoch = %3d, MSE = %8.6f, MAE = %8.6f)'
          % (np.max(val_drug_qed_r2),
             (np.argmax(val_drug_qed_r2) + 1),
             val_drug_qed_mse[np.argmax(val_drug_qed_r2)],
             val_drug_qed_mae[np.argmax(val_drug_qed_r2)]))

    # Print best R2 scores for drug response regressor
    val_data_sources = \
        [loader.dataset.data_source for loader in drug_resp_val_loaders]
    best_r2 = np.amax(val_resp_r2, axis=0)
    best_r2_epochs = np.argmax(val_resp_r2, axis=0)

    for index, data_source in enumerate(val_data_sources):
        print('\t\t%-6s \t Best R2 Score: %+6.4f '
              '(Epoch = %3d, MSE = %8.2f, MAE = %6.2f)'
              % (data_source, best_r2[index],
                 best_r2_epochs[index] + args.resp_val_start_epoch + 1,
                 val_resp_mse[best_r2_epochs[index], index],
                 val_resp_mae[best_r2_epochs[index], index]))

    # Print best epoch and all the corresponding validation results
    # Picking the best epoch using R2 score from same data source
    best_epoch = val_resp_r2[:, val_index].argmax()
    print('\n\tBest Results from the Same Model (Epoch = %3d):'
          % (best_epoch + 1))
    for index, clf_target in enumerate(clf_targets):
        print('\t\t%-24s Accuracy: %.3f%%'
              % (clf_target, val_cl_clf_acc[best_epoch, index]))

    # Print best predictions for drug classifiers and regressor
    print('\t\tDrug Target Family \t Accuracy: %.3f%% '
          % (val_drug_target_acc[best_epoch]))

    print('\t\tDrug Weighted QED \t R2 Score: %+6.4f '
          '(MSE = %8.6f, MAE = %6.6f)'
          % (val_drug_qed_r2[best_epoch],
             val_drug_qed_mse[best_epoch],
             val_drug_qed_mae[best_epoch]))

    for index, data_source in enumerate(val_data_sources):
        print('\t\t%-6s \t R2 Score: %+6.4f '
              '(MSE = %8.2f, MAE = %6.2f)'
              % (data_source,
                 val_resp_r2[best_epoch, index],
                 val_resp_mse[best_epoch, index],
                 val_resp_mae[best_epoch, index]))


# Use ./launcher.py for more convenient calling and logging
main()
