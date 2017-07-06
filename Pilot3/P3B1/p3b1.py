from __future__ import absolute_import
from __future__ import print_function
import os
import sys
import argparse
try:
    import configparser
except ImportError:
    import ConfigParser as configparser

file_path = os.path.dirname(os.path.realpath(__file__))
lib_path = os.path.abspath(os.path.join(file_path, '..', 'common'))
sys.path.append(lib_path)
lib_path2 = os.path.abspath(os.path.join(file_path, '..', '..', 'common'))
sys.path.append(lib_path2)

import p3_common

def common_parser(parser):

    parser.add_argument("--config_file", dest='config_file', type=str,
                        default=os.path.join(file_path, 'p3b1_default_model.txt'),
                        help="specify model configuration file")

    # Parse has been split between arguments that are common with the default neon parser
    # and all the other options
    parser = p3_common.get_default_neon_parse(parser)
    parser = p3_common.get_p3_common_parser(parser)

    # Arguments that are applicable just to p3b1
    parser = p3b1_parser(parser)

    return parser

def p3b1_parser(parser):
    ### Hyperparameters and model save path

    # these are leftover from other models but don't conflict so leave for now
    parser.add_argument("--train", action="store_true",dest="train_bool",default=True,help="Invoke training")
    parser.add_argument("--evaluate", action="store_true",dest="eval_bool",default=False,help="Use model for inference")
    parser.add_argument("--home-dir",help="Home Directory",dest="home_dir",type=str,default='.')
    parser.add_argument("--save-dir",help="Save Directory",dest="save_path",type=str,default=None)
    parser.add_argument("--config-file",help="Config File",dest="config_file",type=str,default=os.path.join(file_path, 'p3b1_default_model.txt'))
    parser.add_argument("--memo",help="Memo",dest="base_memo",type=str,default=None)
    parser.add_argument("--seed", action="store_true",dest="seed",default=False,help="Random Seed")
    parser.add_argument("--case",help="[Full, Center, CenterZ]",dest="case",type=str,default='CenterZ')
    parser.add_argument("--fig", action="store_true",dest="fig_bool",default=False,help="Generate Prediction Figure")

    # MTL_run params start here
    parser.add_argument("-v", "--verbose", action="store_true",
                        default= True,
                        help="increase output verbosity")

    #parser.add_argument("-a", "--activation", action="store",
                        #default=ACTIVATION,
                        #help="keras activation function to use in inner layers: relu, tanh, sigmoid...")
    #parser.add_argument("-b", "--batch_size", action="store",
                        #default=BATCH_SIZE, type=int,
                        #help="batch size")
    #parser.add_argument("-e", "--epochs", action="store",
                        #default=N_EPOCHS, type=int,
                        #help="number of training epochs")
    # parser.add_argument("-o", "--optimizer", action="store",
    #                     default=OPTIMIZER,
    #                     help="keras optimizer to use: sgd, rmsprop, ...")
    parser.add_argument("--dropout", action="store",
                        default=argparse.SUPPRESS, # DROPOUT, type=float,
                        help="ratio of dropout used in fully connected layers")
    parser.add_argument("--learning_rate", action='store',
                        default=argparse.SUPPRESS, #  LEARNING_RATE, type=float,
                        help='learning rate')

    parser.add_argument("--train_features", action="store",
                        default='data/task0_0_train_feature.csv;data/task1_0_train_feature.csv;data/task2_0_train_feature.csv',
                        help='training feature data filenames')
    parser.add_argument("--train_truths", action="store",
                        default='data/task0_0_train_label.csv;data/task1_0_train_label.csv;data/task2_0_train_label.csv',
                        help='training truth data filenames')

    parser.add_argument("--valid_features", action="store",
                        default='data/task0_0_test_feature.csv;data/task1_0_test_feature.csv;data/task2_0_test_feature.csv',
                        help='validation feature data filenames')
    parser.add_argument("--valid_truths", action="store",
                        default='data/task0_0_test_label.csv;data/task1_0_test_label.csv;data/task2_0_test_label.csv',
                        help='validation truth data filenames')

    parser.add_argument("--output_files", action="store",
                        default='result0_0.csv;result1_0.csv;result2_0.csv',
                        help="output filename")

    parser.add_argument("--shared_nnet_spec", action="store",
                        default=argparse.SUPPRESS, # DEF_SHARED_NNET_SPEC,
                        help='network structure of shared layer')
    parser.add_argument("--individual_nnet_spec", action="store",
                        default=argparse.SUPPRESS, # DEF_INDIV_NNET_SPEC,
                        help='network structore of task-specific layer')

    return parser


def read_config_file(File):
    config=configparser.ConfigParser()
    config.read(File)
    section=config.sections()
    Global_Params={}

    Global_Params['learning_rate'] =eval(config.get(section[0],'learning_rate'))
    Global_Params['batch_size']    =eval(config.get(section[0],'batch_size'))
    Global_Params['epochs']        =eval(config.get(section[0],'epochs'))
    Global_Params['dropout']       =eval(config.get(section[0],'dropout'))
    Global_Params['activation']    =eval(config.get(section[0],'activation'))
    Global_Params['out_act']          =eval(config.get(section[0],'out_act'))
    Global_Params['loss']          =eval(config.get(section[0],'loss'))
    Global_Params['n_fold']          =eval(config.get(section[0],'n_fold'))
    Global_Params['optimizer']     =eval(config.get(section[0],'optimizer'))
    Global_Params['shared_nnet_spec']          =eval(config.get(section[0],'shared_nnet_spec'))
    Global_Params['ind_nnet_spec']          =eval(config.get(section[0],'ind_nnet_spec'))
    Global_Params['feature_names']          =eval(config.get(section[0],'feature_names'))

    # note 'cool' is a boolean
    #Global_Params['cool']          =config.get(section[0],'cool')
    return Global_Params
