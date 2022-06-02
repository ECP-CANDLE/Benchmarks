"""
    File Name:          UnoPytorch/uno_pytorch.py
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               8/13/18
    Python Version:     3.6.6
    File Description:   This is a version of the original file
                        modified to fit CANDLE framework.
                        Date: 3/12/19.

"""

import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
from networks.initialization.encoder_init import get_gene_encoder, \
    get_drug_encoder
from utils.datasets.drug_target_dataset import DrugTargetDataset
from utils.miscellaneous.optimizer import get_optimizer


# Number of workers for dataloader. Too many workers might lead to process
# hanging for PyTorch version 4.1. Set this number between 0 and 4.
NUM_WORKER = 4
DATA_ROOT = '../../Data/Pilot1/'


class UnoMTModel(object):

    def __init__(self, args, use_cuda, device):

        self.args = args
        self.use_cuda = use_cuda
        self.device = device
        self.config_data_loaders()
        self.build_data_loaders()
        self.build_nn()
        self.config_optimization()

    def config_data_loaders(self):

        args = self.args

        # Data loaders for training/validation ####################################
        self.dataloader_kwargs = {
            'timeout': 1,
            'shuffle': 'True',
            # 'num_workers': multiprocessing.cpu_count() if use_cuda else 0,
            'num_workers': NUM_WORKER if self.use_cuda else 0,
            'pin_memory': True if self.use_cuda else False, }

        # Drug response dataloaders for training/validation
        self.drug_resp_dataset_kwargs = {
            'data_root': DATA_ROOT,
            'rand_state': args.rng_seed,
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
            'validation_ratio': args.val_split,
            'disjoint_drugs': args.disjoint_drugs,
            'disjoint_cells': args.disjoint_cells, }

        # Cell line classification dataloaders for training/validation
        self.cl_clf_dataset_kwargs = {
            'data_root': DATA_ROOT,
            'rand_state': args.rng_seed,
            'summary': False,

            'int_dtype': np.int8,
            'float_dtype': np.float16,
            'output_dtype': np.float32,

            'rnaseq_scaling': args.rnaseq_scaling,

            'rnaseq_feature_usage': args.rnaseq_feature_usage,
            'validation_ratio': args.val_split, }

        # Drug target family classification dataloaders for training/validation
        self.drug_target_dataset_kwargs = {
            'data_root': DATA_ROOT,
            'rand_state': args.rng_seed,
            'summary': False,

            'int_dtype': np.int8,
            'float_dtype': np.float16,
            'output_dtype': np.float32,

            'dscptr_scaling': args.dscptr_scaling,
            'dscptr_nan_threshold': args.dscptr_nan_threshold,

            'drug_feature_usage': args.drug_feature_usage,
            'validation_ratio': args.val_split, }

        # Drug weighted QED regression dataloaders for training/validation
        self.drug_qed_dataset_kwargs = {
            'data_root': DATA_ROOT,
            'rand_state': args.rng_seed,
            'summary': False,

            'int_dtype': np.int8,
            'float_dtype': np.float16,
            'output_dtype': np.float32,

            'qed_scaling': args.qed_scaling,
            'dscptr_scaling': args.dscptr_scaling,
            'dscptr_nan_threshold': args.dscptr_nan_threshold,

            'drug_feature_usage': args.drug_feature_usage,
            'validation_ratio': args.val_split, }

    def build_data_loaders(self):

        args = self.args

        self.drug_resp_trn_loader = torch.utils.data.DataLoader(
            DrugRespDataset(data_src=args.train_sources,
                            training=True,
                            **(self.drug_resp_dataset_kwargs)),
            batch_size=args.trn_batch_size,
            **(self.dataloader_kwargs))

        # List of data loaders for different validation sets
        self.drug_resp_val_loaders = [torch.utils.data.DataLoader(
            DrugRespDataset(data_src=src,
                            training=False,
                            **(self.drug_resp_dataset_kwargs)),
            batch_size=args.val_batch_size,
            **(self.dataloader_kwargs)) for src in args.val_sources]

        self.cl_clf_trn_loader = torch.utils.data.DataLoader(
            CLClassDataset(training=True,
                           **(self.cl_clf_dataset_kwargs)),
            batch_size=args.trn_batch_size,
            **(self.dataloader_kwargs))

        self.cl_clf_val_loader = torch.utils.data.DataLoader(
            CLClassDataset(training=False,
                           **(self.cl_clf_dataset_kwargs)),
            batch_size=args.val_batch_size,
            **(self.dataloader_kwargs))

        self.drug_target_trn_loader = torch.utils.data.DataLoader(
            DrugTargetDataset(training=True,
                              **(self.drug_target_dataset_kwargs)),
            batch_size=args.trn_batch_size,
            **(self.dataloader_kwargs))

        self.drug_target_val_loader = torch.utils.data.DataLoader(
            DrugTargetDataset(training=False,
                              **(self.drug_target_dataset_kwargs)),
            batch_size=args.val_batch_size,
            **(self.dataloader_kwargs))

        self.drug_qed_trn_loader = torch.utils.data.DataLoader(
            DrugQEDDataset(training=True,
                           **(self.drug_qed_dataset_kwargs)),
            batch_size=args.trn_batch_size,
            **(self.dataloader_kwargs))

        self.drug_qed_val_loader = torch.utils.data.DataLoader(
            DrugQEDDataset(training=False,
                           **(self.drug_qed_dataset_kwargs)),
            batch_size=args.val_batch_size,
            **(self.dataloader_kwargs))

    def build_nn(self):

        args = self.args
        device = self.device

        # Constructing and initializing neural networks ###########################
        # Autoencoder training hyper-parameters
        self.ae_training_kwarg = {
            'ae_loss_func': 'mse',
            'ae_opt': 'sgd',
            'ae_lr': 2e-1,
            'ae_reg': 1e-5,
            'lr_decay_factor': 1.0,
            'max_num_epochs': 1000,
            'early_stop_patience': 50, }

        self.encoder_kwarg = {
            'model_folder': './models/',
            'data_root': DATA_ROOT,

            'autoencoder_init': args.autoencoder_init,
            'training_kwarg': self.ae_training_kwarg,

            'device': device,
            'verbose': True,
            'rand_state': args.rng_seed, }

        # Get RNA sequence encoder
        self.gene_encoder = get_gene_encoder(
            rnaseq_feature_usage=args.rnaseq_feature_usage,
            rnaseq_scaling=args.rnaseq_scaling,

            layer_dim=args.gene_layer_dim,
            num_layers=args.gene_num_layers,
            latent_dim=args.gene_latent_dim,
            **(self.encoder_kwarg))

        # Get drug feature encoder
        self.drug_encoder = get_drug_encoder(
            drug_feature_usage=args.drug_feature_usage,
            dscptr_scaling=args.dscptr_scaling,
            dscptr_nan_threshold=args.dscptr_nan_threshold,

            layer_dim=args.drug_layer_dim,
            num_layers=args.drug_num_layers,
            latent_dim=args.drug_latent_dim,
            **(self.encoder_kwarg))

        # Regressor for drug response
        self.resp_net = RespNet(
            gene_latent_dim=args.gene_latent_dim,
            drug_latent_dim=args.drug_latent_dim,

            gene_encoder=self.gene_encoder,
            drug_encoder=self.drug_encoder,

            resp_layer_dim=args.resp_layer_dim,
            resp_num_layers_per_block=args.resp_num_layers_per_block,
            resp_num_blocks=args.resp_num_blocks,
            resp_num_layers=args.resp_num_layers,
            resp_dropout=args.dropout,

            resp_activation=args.resp_activation).to(device)

        print(self.resp_net)

        # Sequence classifier for category, site, and type
        self.cl_clf_net_kwargs = {
            'encoder': self.gene_encoder,
            'input_dim': args.gene_latent_dim,
            'condition_dim': len(get_label_dict(DATA_ROOT, 'data_src_dict.txt')),
            'layer_dim': args.cl_clf_layer_dim,
            'num_layers': args.cl_clf_num_layers, }

        self.category_clf_net = ClfNet(
            num_classes=len(get_label_dict(DATA_ROOT, 'category_dict.txt')),
            **(self.cl_clf_net_kwargs)).to(device)
        self.site_clf_net = ClfNet(
            num_classes=len(get_label_dict(DATA_ROOT, 'site_dict.txt')),
            **(self.cl_clf_net_kwargs)).to(device)
        self.type_clf_net = ClfNet(
            num_classes=len(get_label_dict(DATA_ROOT, 'type_dict.txt')),
            **(self.cl_clf_net_kwargs)).to(device)

        # Classifier for drug target family prediction
        self.drug_target_net = ClfNet(
            encoder=self.drug_encoder,
            input_dim=args.drug_latent_dim,
            condition_dim=0,
            layer_dim=args.drug_target_layer_dim,
            num_layers=args.drug_target_num_layers,
            num_classes=len(get_label_dict(DATA_ROOT, 'drug_target_dict.txt'))).\
            to(device)

        # Regressor for drug weighted QED prediction
        self.drug_qed_net = RgsNet(
            encoder=self.drug_encoder,
            input_dim=args.drug_latent_dim,
            condition_dim=0,
            layer_dim=args.drug_qed_layer_dim,
            num_layers=args.drug_qed_num_layers,
            activation=args.drug_qed_activation).to(device)

        # Multi-GPU settings
        if args.multi_gpu:
            self.resp_net = nn.DataParallel(self.resp_net)
            self.category_clf_net = nn.DataParallel(self.category_clf_net)
            self.site_clf_net = nn.DataParallel(self.site_clf_net)
            self.type_clf_net = nn.DataParallel(self.type_clf_net)
            self.drug_target_net = nn.DataParallel(self.drug_target_net)
            self.drug_qed_net = nn.DataParallel(self.drug_qed_net)

    def config_optimization(self):

        args = self.args

        # Optimizers, learning rate decay, and miscellaneous ######################
        self.update_l2regularizer(args.l2_regularization)

        self.resp_lr_decay = LambdaLR(optimizer=self.resp_opt,
                                      lr_lambda=lambda e:
                                      args.lr_decay_factor ** e)
        self.cl_clf_lr_decay = LambdaLR(optimizer=self.cl_clf_opt,
                                        lr_lambda=lambda e:
                                        args.lr_decay_factor ** e)
        self.drug_target_lr_decay = LambdaLR(optimizer=self.drug_target_opt,
                                             lr_lambda=lambda e:
                                             args.lr_decay_factor ** e)
        self.drug_qed_lr_decay = LambdaLR(optimizer=self.drug_qed_opt,
                                          lr_lambda=lambda e:
                                          args.lr_decay_factor ** e)

        self.resp_loss_func = F.l1_loss if args.resp_loss_func == 'l1' \
            else F.mse_loss
        self.drug_qed_loss_func = F.l1_loss if args.drug_qed_loss_func == 'l1' \
            else F.mse_loss

    def update_l2regularizer(self, reg):

        args = self.args

        self.resp_opt = get_optimizer(opt_type=args.resp_opt,
                                      networks=self.resp_net,
                                      learning_rate=args.resp_lr,
                                      l2_regularization=reg)

        self.cl_clf_opt = get_optimizer(opt_type=args.cl_clf_opt,
                                        networks=[self.category_clf_net,
                                                  self.site_clf_net,
                                                  self.type_clf_net],
                                        learning_rate=self.args.cl_clf_lr,
                                        l2_regularization=reg)

        self.drug_target_opt = get_optimizer(opt_type=args.drug_target_opt,
                                             networks=self.drug_target_net,
                                             learning_rate=args.drug_target_lr,
                                             l2_regularization=reg)

        self.drug_qed_opt = get_optimizer(opt_type=args.drug_qed_opt,
                                          networks=self.drug_qed_net,
                                          learning_rate=args.drug_qed_lr,
                                          l2_regularization=reg)

    def update_dropout(self, dropout_rate):

        self.args.dropout = dropout_rate

        # Regressor for drug response
        self.resp_net = RespNet(
            gene_latent_dim=self.args.gene_latent_dim,
            drug_latent_dim=self.args.drug_latent_dim,

            gene_encoder=self.gene_encoder,
            drug_encoder=self.drug_encoder,

            resp_layer_dim=self.args.resp_layer_dim,
            resp_num_layers_per_block=self.args.resp_num_layers_per_block,
            resp_num_blocks=self.args.resp_num_blocks,
            resp_num_layers=self.args.resp_num_layers,
            resp_dropout=self.args.dropout,


            resp_activation=self.args.resp_activation).to(self.device)

    def pre_train_config(self):

        print('Data sizes:\nTrain:')
        print('Data set: ' + self.drug_resp_trn_loader.dataset.data_source + ' Size: ' + str(len(self.drug_resp_trn_loader.dataset)))
        print('Validation:')

        # Early stopping is decided on the validation set with the same
        # data source as the training dataloader
        self.val_index = 0
        for idx, loader in enumerate(self.drug_resp_val_loaders):
            print('Data set: ' + loader.dataset.data_source + ' Size: ' + str(len(loader.dataset)))
            if loader.dataset.data_source == self.args.train_sources:
                self.val_index = idx

    def train(self):

        args = self.args
        device = self.device

        # Training/validation loops ###############################################
        self.val_cl_clf_acc = []
        self.val_drug_target_acc = []
        self.val_drug_qed_mse = []
        self.val_drug_qed_mae = []
        self.val_drug_qed_r2 = []
        self.val_resp_mse = []
        self.val_resp_mae = []
        self.val_resp_r2 = []
        self.best_r2 = -np.inf
        self.patience = 0
        self.start_time = time.time()

        for epoch in range(args.epochs):

            print('=' * 80 + '\nTraining Epoch %3i:' % (epoch + 1))
            epoch_start_time = time.time()

            self.resp_lr_decay.step(epoch)
            self.cl_clf_lr_decay.step(epoch)
            self.drug_target_lr_decay.step(epoch)
            self.drug_qed_lr_decay.step(epoch)

            # Training cell line classifier
            train_cl_clf(device=device,
                         category_clf_net=self.category_clf_net,
                         site_clf_net=self.site_clf_net,
                         type_clf_net=self.type_clf_net,
                         data_loader=self.cl_clf_trn_loader,
                         max_num_batches=args.max_num_batches,
                         optimizer=self.cl_clf_opt)

            # Training drug target classifier
            train_drug_target(device=device,
                              drug_target_net=self.drug_target_net,
                              data_loader=self.drug_target_trn_loader,
                              max_num_batches=args.max_num_batches,
                              optimizer=self.drug_target_opt)

            # Training drug weighted QED regressor
            train_drug_qed(device=device,
                           drug_qed_net=self.drug_qed_net,
                           data_loader=self.drug_qed_trn_loader,
                           max_num_batches=args.max_num_batches,
                           loss_func=self.drug_qed_loss_func,
                           optimizer=self.drug_qed_opt)

            # Training drug response regressor
            train_resp(device=device,
                       resp_net=self.resp_net,
                       data_loader=self.drug_resp_trn_loader,
                       max_num_batches=args.max_num_batches,
                       loss_func=self.resp_loss_func,
                       optimizer=self.resp_opt)

            if epoch >= args.resp_val_start_epoch:

                resp_r2 = self.validation(epoch)

                # Record the best R2 score (same data source)
                # and check for early stopping if no improvement for epochs
                if resp_r2[self.val_index] > self.best_r2:
                    self.patience = 0
                    self.best_r2 = resp_r2[self.val_index]
                else:
                    self.patience += 1
                if self.patience >= args.early_stop_patience:
                    print('Validation results does not improve for %d epochs ... '
                          'invoking early stopping.' % self.patience)
                    break

            print('Epoch Running Time: %.1f Seconds.'
                  % (time.time() - epoch_start_time))

    def validation(self, epoch):

        # args = self.args
        device = self.device

        # Validating cell line classifier
        cl_category_acc, cl_site_acc, cl_type_acc = \
            valid_cl_clf(device=device,
                         category_clf_net=self.category_clf_net,
                         site_clf_net=self.site_clf_net,
                         type_clf_net=self.type_clf_net,
                         data_loader=self.cl_clf_val_loader, )

        self.val_cl_clf_acc.append([cl_category_acc, cl_site_acc, cl_type_acc])

        # Validating drug target classifier
        drug_target_acc = \
            valid_drug_target(device=device,
                              drug_target_net=self.drug_target_net,
                              data_loader=self.drug_target_val_loader)
        self.val_drug_target_acc.append(drug_target_acc)

        # Validating drug weighted QED regressor
        drug_qed_mse, drug_qed_mae, drug_qed_r2 = \
            valid_drug_qed(device=device,
                           drug_qed_net=self.drug_qed_net,
                           data_loader=self.drug_qed_val_loader)

        self.val_drug_qed_mse.append(drug_qed_mse)
        self.val_drug_qed_mae.append(drug_qed_mae)
        self.val_drug_qed_r2.append(drug_qed_r2)

        # Validating drug response regressor
        resp_mse, resp_mae, resp_r2 = \
            valid_resp(device=device,
                       resp_net=self.resp_net,
                       data_loaders=self.drug_resp_val_loaders)

        # Save the validation results in nested list
        self.val_resp_mse.append(resp_mse)
        self.val_resp_mae.append(resp_mae)
        self.val_resp_r2.append(resp_r2)

        return resp_r2

    def print_final_stats(self):

        args = self.args

        val_cl_clf_acc = np.array(self.val_cl_clf_acc).reshape(-1, 3)
        # val_drug_target_acc = np.array(val_drug_target_acc)
        # val_drug_qed_mse = np.array(val_drug_qed_mse)
        # val_resp_mae = np.array(val_resp_mae)
        # val_resp_r2 = np.array(val_resp_r2)
        val_resp_mse, val_resp_mae, val_resp_r2 = \
            np.array(self.val_resp_mse).reshape(-1, len(args.val_sources)), \
            np.array(self.val_resp_mae).reshape(-1, len(args.val_sources)), \
            np.array(self.val_resp_r2).reshape(-1, len(args.val_sources))

        print('Program Running Time: %.1f Seconds.' % (time.time() - self.start_time))

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
                  % (clf_target, best_acc[index],
                     best_acc_epochs[index] + 1 + args.resp_val_start_epoch))

        # Print best predictions for drug classifiers and regressor
        print('\t\tDrug Target Family \t Best Accuracy: %.3f%% (Epoch = %3d)'
              % (np.max(self.val_drug_target_acc),
                 (np.argmax(self.val_drug_target_acc) + 1 + args.resp_val_start_epoch)))

        print('\t\tDrug Weighted QED \t Best R2 Score: %+6.4f '
              '(Epoch = %3d, MSE = %8.6f, MAE = %8.6f)'
              % (np.max(self.val_drug_qed_r2),
                 (np.argmax(self.val_drug_qed_r2) + 1 + args.resp_val_start_epoch),
                  self.val_drug_qed_mse[np.argmax(self.val_drug_qed_r2)],
                  self.val_drug_qed_mae[np.argmax(self.val_drug_qed_r2)]))

        # Print best R2 scores for drug response regressor
        val_data_sources = \
            [loader.dataset.data_source for loader in self.drug_resp_val_loaders]
        best_r2 = np.amax(self.val_resp_r2, axis=0)
        best_r2_epochs = np.argmax(self.val_resp_r2, axis=0)

        for index, data_source in enumerate(val_data_sources):
            print('\t\t%-6s \t Best R2 Score: %+6.4f '
                  '(Epoch = %3d, MSE = %8.2f, MAE = %6.2f)'
                  % (data_source, best_r2[index],
                     best_r2_epochs[index] + args.resp_val_start_epoch + 1,
                     val_resp_mse[best_r2_epochs[index], index],
                     val_resp_mae[best_r2_epochs[index], index]))

        # Print best epoch and all the corresponding validation results
        # Picking the best epoch using R2 score from same data source
        best_epoch = val_resp_r2[:, self.val_index].argmax()
        print('\n\tBest Results from the Same Model (Epoch = %3d):'
              % (best_epoch + 1 + args.resp_val_start_epoch))
        for index, clf_target in enumerate(clf_targets):
            print('\t\t%-24s Accuracy: %.3f%%'
                  % (clf_target, val_cl_clf_acc[best_epoch, index]))

        # Print best predictions for drug classifiers and regressor
        print('\t\tDrug Target Family \t Accuracy: %.3f%% '
              % (self.val_drug_target_acc[best_epoch]))

        print('\t\tDrug Weighted QED \t R2 Score: %+6.4f '
              '(MSE = %8.6f, MAE = %6.6f)'
              % (self.val_drug_qed_r2[best_epoch],
                 self.val_drug_qed_mse[best_epoch],
                 self.val_drug_qed_mae[best_epoch]))

        for index, data_source in enumerate(val_data_sources):
            print('\t\t%-6s \t R2 Score: %+6.4f '
                  '(MSE = %8.2f, MAE = %6.2f)'
                  % (data_source,
                     val_resp_r2[best_epoch, index],
                     val_resp_mse[best_epoch, index],
                     val_resp_mae[best_epoch, index]))
