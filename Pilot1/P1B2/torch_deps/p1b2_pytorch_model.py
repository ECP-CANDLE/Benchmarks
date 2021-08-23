import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.core.multiarray import ndarray
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR

#from utils.optimizer import get_optimizer
from torch_deps.random_seeding import seed_random_state
from torch_deps.p1b2_dataset_pytorch import P1B2Dataset
from torch_deps.p1b2_classification_net import P1B2Net
from pytorch_utils import build_optimizer
from torch_deps.p1b2_clf_func import train_p1b2_clf, valid_p1b2_clf
from torch.utils.tensorboard import SummaryWriter

# Number of workers for dataloader. Too many workers might lead to process
# hanging for PyTorch version 4.1. Set this number between 0 and 4.
NUM_WORKER = 4
DATA_ROOT = '../../Data/Pilot1/'


class P1B2Model(object):

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
        self.p1b2_dataset_kwargs = {
            'data_root': DATA_ROOT,
            'rand_state': args.rng_seed,
            'summary': True,

            'data_url': args.data_url,
            'train_data': args.train_data,
            'test_data': args.test_data,
            'feature_subsample': args.feature_subsample,
            'classes': args.classes,
            'data_type': args.data_type,
            'shuffle': args.shuffle,
            'val_split': args.val_split,

            'int_dtype': np.int8,
            'float_dtype': np.float16,
            'output_dtype': np.float32,

            'feature_subsample': args.feature_subsample, }


    def build_data_loaders(self):

        args = self.args

        self.p1b2_trn_loader = torch.utils.data.DataLoader(
                P1B2Dataset(data_src=args.train_data,
                        training=1,
                        **(self.p1b2_dataset_kwargs)),
                batch_size=args.batch_size,
                **(self.dataloader_kwargs))

        self.args.input_dim = P1B2Dataset.input_dim

        # Data loader for validation
        self.p1b2_val_loader = torch.utils.data.DataLoader(
                P1B2Dataset(data_src=args.test_data,training=2,
                       **(self.p1b2_dataset_kwargs)),
                batch_size=args.batch_size,
                **(self.dataloader_kwargs))

        # Data loader for test
        self.p1b2_test_loader = torch.utils.data.DataLoader(
                P1B2Dataset(data_src=args.test_data,training=0,
                       **(self.p1b2_dataset_kwargs)),
                batch_size=args.batch_size,
                **(self.dataloader_kwargs))


    def build_nn(self):

        args = self.args
        device = self.device
        #args.input_dim = [1, 28204]

        # Sequence classifier
        self.p1b2_net_kwargs = {

            'layers': args.dense,
            'activation': args.activation,
            'out_activation': args.out_activation,
            'dropout': args.dropout,
            #'classes': args.classes,
            'input_dim': args.input_dim, }

        self.p1b2_net = P1B2Net(
                        classes=args.classes,
                        **(self.p1b2_net_kwargs)).to(device)

        # Multi-GPU settings
        if self.use_cuda and args.multi_gpu:
            self.p1b2_net = nn.DataParallel(self.p1b2_net)


    def config_optimization(self):

        args = self.args
        weight_decay = args.reg_l2

        type = args.optimizer
        lr = args.learning_rate
        #kerasDefaults = {}
        #kerasDefaults['momentum_sgd'] = 0
        #kerasDefaults['nesterov_sgd'] = False
        kerasDefaults = args.keras_defaults
        kerasDefaults['weight_decay'] = weight_decay

        self.p1b2_optimizer = build_optimizer(self.p1b2_net, type, lr, kerasDefaults, trainable_only=False)

        #reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=1, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)
        self.p1b2_scheduler = ReduceLROnPlateau(self.p1b2_optimizer, mode='min', factor=0.1, patience=10, verbose=True, eps=0.0001, cooldown=0, min_lr=0)


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

            curr_lr = self.p1b2_optimizer.param_groups[0]['lr']

            # Training Disease Type classifier
            train_acc,train_loss = train_p1b2_clf(device=device,
                     category_clf_net=self.p1b2_net,
                     data_loader=self.p1b2_trn_loader,
                     loss_type=args.loss,
                     max_num_batches=args.max_num_batches,
                     optimizer=self.p1b2_optimizer,
                     scheduler=self.p1b2_scheduler,
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
                
                if hasattr(self.p1b2_net,'module'):
                    # Saving the DataParallel model
                    torch.save(self.p1b2_net.module.state_dict(), args.model_autosave_filename)
                else:
                    torch.save(self.p1b2_net.state_dict(), args.model_autosave_filename)
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
            self.p1b2_scheduler.step(val_loss)

            print('Epoch Running Time: %.1f Seconds.'
              % (time.time() - epoch_start_time))

        # close the csv logger file
        csv_logger_f.close()
        tensorboard_writer.flush()
        tensorboard_writer.close()

    def validation(self, epoch):

        args = self.args
        device = self.device


        # Validating Disease Type Classifier
        cl_category_acc, category_loss = \
            valid_p1b2_clf(device=device,
                             category_clf_net=self.p1b2_net,
                             data_loader=self.p1b2_val_loader,
                             loss_type=args.loss, )

        self.val_cl_clf_acc.append([cl_category_acc, category_loss])

        return cl_category_acc,category_loss

    def test(self):

        args = self.args
        device = self.device


        # Test Disease Type Classifier
        cl_category_acc, category_loss = \
            valid_p1b2_clf(device=device,
                             category_clf_net=self.p1b2_net,
                             data_loader=self.p1b2_test_loader,
                             loss_type=args.loss, )

        #self.val_cl_clf_acc.append([cl_category_acc, category_loss])

        return cl_category_acc,category_loss

    def print_final_stats(self):

        args = self.args

        val_cl_clf_acc = np.array(self.val_cl_clf_acc).reshape(-1, 2)

        print('Test data: ')
        test_acc,test_loss = self.test()

        print('Program Running Time: %.1f Seconds.' % (time.time() - self.start_time))

        # Print overall validation results
        print('=' * 80)
        print('Overall Validation Results:\n')

        print('\tBest Results from P1B2 Models (Epochs):')
        # Print best accuracy for Disease Type classifiers
        clf_targets = ['Disease Type Classifier from Somatic SNPs',
                      ]
        best_acc = np.amax(val_cl_clf_acc, axis=0)
        best_acc_epochs = np.argmax(val_cl_clf_acc, axis=0)

        for index, clf_target in enumerate(clf_targets):
            print('\t\t%-24s Best Accuracy: %.3f%% (Epoch = %3d)'
              % (clf_target, best_acc[index],
                 best_acc_epochs[index] + 1 ))

        print('=' * 80)
        print('Test Results:  Accuracy: %.3f%%  Loss=%5.5f\n'
                %(test_acc, test_loss))
        print('=' * 80)