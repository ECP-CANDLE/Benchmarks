from __future__ import print_function

import numpy as np

from sklearn.metrics import accuracy_score

import os
import sys
#import logging
import argparse

try:
    import configparser
except ImportError:
    import ConfigParser as configparser

file_path = os.path.dirname(os.path.realpath(__file__))
lib_path = os.path.abspath(os.path.join(file_path, '..'))
sys.path.append(lib_path)
lib_path2 = os.path.abspath(os.path.join(file_path, '..', '..', 'common'))
sys.path.append(lib_path2)

import default_utils
import data_utils


class BenchmarkP3B1(default_utils.Benchmark):


    def parse_from_benchmark(self):

        self.parser.add_argument("--train_features", action="store",
                        default='data/task0_0_train_feature.csv;data/task1_0_train_feature.csv;data/task2_0_train_feature.csv',
                        help='training feature data filenames')
        self.parser.add_argument("--train_truths", action="store",
                        default='data/task0_0_train_label.csv;data/task1_0_train_label.csv;data/task2_0_train_label.csv',
                        help='training truth data filenames')

        self.parser.add_argument("--valid_features", action="store",
                        default='data/task0_0_test_feature.csv;data/task1_0_test_feature.csv;data/task2_0_test_feature.csv',
                        help='validation feature data filenames')
        self.parser.add_argument("--valid_truths", action="store",
                        default='data/task0_0_test_label.csv;data/task1_0_test_label.csv;data/task2_0_test_label.csv',
                        help='validation truth data filenames')

        self.parser.add_argument("--output_files", action="store",
                        default='result0_0.csv;result1_0.csv;result2_0.csv',
                        help="output filename")

        self.parser.add_argument("--shared_nnet_spec", action="store",
                        default=argparse.SUPPRESS, # DEF_SHARED_NNET_SPEC,
                        help='network structure of shared layer')
        self.parser.add_argument("--individual_nnet_spec", action="store",
                        default=argparse.SUPPRESS, # DEF_INDIV_NNET_SPEC,
                        help='network structore of task-specific layer')
    
        self.parser.add_argument("--case",help="[Full, Center, CenterZ]",dest="case",type=str,default='CenterZ')
        self.parser.add_argument("--fig", action="store_true",dest="fig_bool",default=False,help="Generate Prediction Figure")
    
                        
    

    def read_config_file(self, file):
        """Functionality to read the configue file
           specific for each benchmark.
        """

        config=configparser.ConfigParser()
        config.read(file)
        section=config.sections()
        fileParams={}
    
        fileParams['activation']=eval(config.get(section[0],'activation'))
        fileParams['batch_size']=eval(config.get(section[0],'batch_size'))
        fileParams['drop'] = eval(config.get(section[0],'drop'))
        fileParams['epochs']=eval(config.get(section[0],'epochs'))
        fileParams['initialization']=eval(config.get(section[0],'initialization'))
        fileParams['learning_rate']=eval(config.get(section[0], 'learning_rate'))
        fileParams['loss']=eval(config.get(section[0],'loss'))
        fileParams['metrics'] = eval(config.get(section[0],'metrics'))        
        fileParams['n_fold'] = eval(config.get(section[0],'n_fold'))
        fileParams['optimizer']=eval(config.get(section[0],'optimizer'))
        fileParams['scaling']=eval(config.get(section[0],'scaling'))
    
        fileParams['data_url']=eval(config.get(section[0],'data_url'))
        fileParams['train_data']=eval(config.get(section[0],'train_data'))
        fileParams['model_name']=eval(config.get(section[0],'model_name'))

        fileParams['shared_nnet_spec']=eval(config.get(section[0],'shared_nnet_spec'))
        fileParams['ind_nnet_spec']=eval(config.get(section[0],'ind_nnet_spec'))
        fileParams['feature_names']=eval(config.get(section[0],'feature_names'))
        #fileParams['cool']=eval(config.get(section[0],'cool'))
        

        # parse the remaining values
        for k,v in config.items(section[0]):
            if not k in fileParams:
                fileParams[k] = eval(v)
    
        return fileParams


def build_data(nnet_spec_len, fold, data_path):
    """ Build feature sets to match the network topology
    """
    X_train = []
    Y_train = []

    X_test = []
    Y_test = []

    for i in range( nnet_spec_len ):
        feature_train = np.genfromtxt(data_path + '/task'+str(i)+'_'+str(fold)+'_train_feature.csv', delimiter= ',' )
        label_train = np.genfromtxt(data_path + '/task'+str(i)+'_'+str(fold)+'_train_label.csv', delimiter= ',' )
        X_train.append( feature_train )
        Y_train.append( label_train )

        feature_test = np.genfromtxt(data_path + '/task'+str(i)+'_'+str(fold)+'_test_feature.csv', delimiter= ',' )
        label_test = np.genfromtxt(data_path + '/task'+str(i)+'_'+str(fold)+'_test_label.csv', delimiter= ',' )
        X_test.append( feature_test )
        Y_test.append( label_test )

    return X_train, Y_train, X_test, Y_test

