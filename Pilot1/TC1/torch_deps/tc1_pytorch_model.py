import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.core.multiarray import ndarray
from torch.optim.lr_scheduler import ReduceLROnPlateau

#from utils.optimizer import get_optimizer
from torch_deps.random_seeding import seed_random_state
from torch_deps.tc1_dataset_pytorch import TC1Dataset
from torch_deps.classification_net import Tc1Net
from pytorch_utils import build_optimizer
from torch_deps.tc1_clf_func import train_tc1_clf, valid_tc1_clf
from torch.utils.tensorboard import SummaryWriter

# Number of workers for dataloader. Too many workers might lead to process
# hanging for PyTorch version 4.1. Set this number between 0 and 4.
NUM_WORKER = 4
DATA_ROOT = '../../Data/Pilot1/'


class TC1Model(object):

    def __init__(self, args, use_cuda=False, device=torch.device('cpu')):

        self.args = args
        self.use_cuda = use_cuda
        self.device = device
        self.val_cl_clf_acc = []
        self.config_data_loaders()
        self.build_data_loaders()
        self.build_nn()
        self.config_optimization()


    def config_data_loaders(self):

        args = self.args


        # Data loaders for training/validation ####################################
        self.dataloader_kwargs = {
            'timeout': 0,
            'shuffle': 'False',
            # 'num_workers': multiprocessing.cpu_count() if use_cuda else 0,
            'num_workers': NUM_WORKER if self.use_cuda else 0,
            'pin_memory': True if self.use_cuda else False, }

        # Drug response dataloaders for training/validation
        self.tc1_dataset_kwargs = {
            'data_root': DATA_ROOT,
            'rand_state': args.rng_seed,
            'summary': True,

            'data_url': args.data_url,
            'train_data': args.train_data,
            'test_data': args.test_data,
            'feature_subsample': args.feature_subsample,
            'classes': args.classes,
            'data_type': args.data_type,

            'int_dtype': np.int8,
            'float_dtype': np.float16,
            'output_dtype': np.float32,

            'feature_subsample': args.feature_subsample, }


    def build_data_loaders(self):

        args = self.args

        self.tc1_trn_loader = torch.utils.data.DataLoader(
                TC1Dataset(data_src=args.train_data,
                        training=True,
                        **(self.tc1_dataset_kwargs)),
                batch_size=args.batch_size,
                **(self.dataloader_kwargs))

        self.args.input_dim = TC1Dataset.input_dim

        # Data loader for  validation
        self.tc1_val_loader = torch.utils.data.DataLoader(
                TC1Dataset(data_src=args.test_data,training=False,
                       **(self.tc1_dataset_kwargs)),
                batch_size=args.batch_size,
                **(self.dataloader_kwargs))


    def build_nn(self):

        args = self.args
        device = self.device
        #args.input_dim = [1, 60483]

        # Sequence classifier
        self.tc1_net_kwargs = {

            'conv': args.conv,
            'dense': args.dense,
            'activation': args.activation,
            'out_activation': args.out_activation,
            'dropout': args.dropout,
            #'classes': args.classes,
            'pool': args.pool if hasattr(args,'pool') else False,
            'locally_connected': args.locally_connected if hasattr(args,'locally_connected') else False,
            'input_dim': args.input_dim, }

        self.tc1_net = Tc1Net(
                        classes=args.classes,
                        **(self.tc1_net_kwargs)).to(device)

        # Multi-GPU settings
        if self.use_cuda and args.multi_gpu:
            self.tc1_net = nn.DataParallel(self.tc1_net)


    def config_optimization(self):

        args = self.args

        type = args.optimizer
        lr = 0.01
        kerasDefaults = {}
        kerasDefaults['momentum_sgd'] = 0
        kerasDefaults['nesterov_sgd'] = False
        self.tc1_optimizer = build_optimizer(self.tc1_net, type, lr, kerasDefaults, trainable_only=False)

        #reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=1, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)
        self.tc1_scheduler = ReduceLROnPlateau(self.tc1_optimizer, mode='min', factor=0.1, patience=10, verbose=True, eps=0.0001, cooldown=0, min_lr=0)


    def train(self):

        args = self.args
        device = self.device

        # Writer will output to ./runs/ directory by default
        tensorboard_writer = SummaryWriter(args.tensorboard_dir)

        # Training/validation loops ###############################################
        self.val_cl_clf_acc = []

        # CSV Logger ##############################################################
        import csv 
        # open the file in the write mode
        csv_logger_f = open(args.csv_filename, 'w', encoding='UTF8', newline='')
        # create the csv writer
        csv_logger = csv.writer(csv_logger_f)
        csv_header = ['epoch', 'lr', 'train_accuracy', 'train_loss', 'val_accuracy', 'val_loss']
        # write the header
        csv_logger.writerow(csv_header)
        args.logger.info('{}'.format(csv_header))
        ############################################################################

        self.start_time = time.time()


        for epoch in range(args.epochs):

            print('=' * 80 + '\nTraining Epoch %3i:' % (epoch + 1))
            epoch_start_time = time.time()

            curr_lr = self.tc1_optimizer.param_groups[0]['lr']

            # Training RNA-seq gene expression classifier
            train_acc,train_loss = train_tc1_clf(device=device,
                     category_clf_net=self.tc1_net,
                     data_loader=self.tc1_trn_loader,
                     loss_type=args.loss,
                     max_num_batches=args.max_num_batches,
                     optimizer=self.tc1_optimizer,
                     scheduler=self.tc1_scheduler,
                     epoch=epoch,
                     log_interval=args.log_interval,
                     dry_run=args.dry_run)

            # Min validation loss until epock - 1
            val_cl_clf_acc = np.array(self.val_cl_clf_acc).reshape(-1, 2)
            min_val_loss = np.amin(val_cl_clf_acc, axis=0)[1] if (epoch > 0) else 1e8

            val_acc,val_loss = self.validation(epoch)

            # Save the model if val_loss has improved 
            if val_loss < min_val_loss:
                print('Train Epoch %3d: val_loss improved from %5.5f to %5.5f, saving model to %s'
                % (epoch+1, min_val_loss, val_loss, args.model_autosave_filename))
                
                if hasattr(self.tc1_net,'module'):
                    # Saving the DataParallel model
                    torch.save(self.tc1_net.module.state_dict(), args.model_autosave_filename)
                else:
                    torch.save(self.tc1_net.state_dict(), args.model_autosave_filename)
            else:
                print('Train Epoch %3d: val_loss did not improve from %5.5f'
                % (epoch+1, min_val_loss))             

            # CSV logger 
            csv_data = [epoch, curr_lr, train_acc, train_loss, val_acc, val_loss]
            csv_logger.writerow(csv_data)
            args.logger.info('{}'.format(csv_data))

            # Tensorboard logger
            tensorboard_writer.add_scalar("Loss/train", train_loss, epoch)
            tensorboard_writer.add_scalar("Loss/val", val_loss, epoch)
            tensorboard_writer.add_scalar("Accuracy/train", train_acc/100, epoch)
            tensorboard_writer.add_scalar("Accuracy/val", val_acc/100, epoch) 
            tensorboard_writer.add_scalar("Train/lr", curr_lr, epoch) 

            # Run the scheduler
            #self.tc1_scheduler.step(torch.as_tensor(val_loss))
            self.tc1_scheduler.step(val_loss)

            print('Epoch Running Time: %.1f Seconds.'
              % (time.time() - epoch_start_time))

        # close the csv logger file
        csv_logger_f.close()
        tensorboard_writer.flush()
        tensorboard_writer.close()

    def validation(self, epoch):

        args = self.args
        device = self.device


        # Validating RNA-seq gene expression classifier
        cl_category_acc, category_loss = \
            valid_tc1_clf(device=device,
                             category_clf_net=self.tc1_net,
                             data_loader=self.tc1_val_loader,
                             loss_type=args.loss, )

        self.val_cl_clf_acc.append([cl_category_acc, category_loss])

        return cl_category_acc,category_loss


    def print_final_stats(self):

        args = self.args

        val_cl_clf_acc = np.array(self.val_cl_clf_acc).reshape(-1, 2)

        print('Program Running Time: %.1f Seconds.' % (time.time() - self.start_time))

        # Print overall validation results
        print('=' * 80)
        print('Overall Validation Results:\n')

        print('\tBest Results from TC1 Models (Epochs):')
        # Print best accuracy for RNA-seq gene expression classifiers
        clf_targets = ['RNA-seq gene expression Categories',
                      ]
        best_acc = np.amax(val_cl_clf_acc, axis=0)
        best_acc_epochs = np.argmax(val_cl_clf_acc, axis=0)

        for index, clf_target in enumerate(clf_targets):
            print('\t\t%-24s Best Accuracy: %.3f%% (Epoch = %3d)'
              % (clf_target, best_acc[index],
                 best_acc_epochs[index] + 1 ))
