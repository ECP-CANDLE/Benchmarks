from __future__ import print_function

import numpy as np

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

#logger = logging.getLogger(__name__)

CLASSES = 2

class BenchmarkNT3(default_utils.Benchmark):


    def parse_from_benchmark(self):

        self.parser.add_argument("--classes", action="store",
                        default= CLASSES, type= int,
                        help= 'number of classes in problem')
    

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
        fileParams['classes'] = eval(config.get(section[0],'classes'))
        fileParams['conv']=eval(config.get(section[0],'conv'))
        fileParams['dense']=eval(config.get(section[0],'dense'))
        fileParams['drop'] = eval(config.get(section[0],'drop'))
        fileParams['epochs']=eval(config.get(section[0],'epochs'))
        fileParams['initialization']=eval(config.get(section[0],'initialization'))
        fileParams['learning_rate']=eval(config.get(section[0], 'learning_rate'))
        fileParams['loss']=eval(config.get(section[0],'loss'))
        fileParams['metrics'] = eval(config.get(section[0],'metrics'))
        fileParams['optimizer']=eval(config.get(section[0],'optimizer'))
        fileParams['out_activation'] = eval(config.get(section[0],'out_activation'))
        fileParams['pool']=eval(config.get(section[0],'pool'))
        fileParams['scaling']=eval(config.get(section[0],'scaling'))
    
        fileParams['data_url']=eval(config.get(section[0],'data_url'))
        fileParams['train_data']=eval(config.get(section[0],'train_data'))
        fileParams['test_data']=eval(config.get(section[0],'test_data'))
        fileParams['model_name']=eval(config.get(section[0],'model_name'))
        fileParams['output_dir'] = eval(config.get(section[0], 'output_dir'))
        

        # parse the remaining values
        for k,v in config.items(section[0]):
            if not k in fileParams:
                fileParams[k] = eval(v)
    
        return fileParams



def load_data(train_file, test_file, params):
    return data_utils.load_Xy_data_noheader(train_file, test_file,
                                params['classes'],
                                scaling=params['scaling'],
                                dtype=params['datatype'])

