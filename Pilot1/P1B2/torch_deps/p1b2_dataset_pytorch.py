""" 
Implements the dataset for P1B2
""" 
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
import p1b2 as bmk

class P1B2Dataset(data.Dataset):

    input_dim = []
    __is_loaded = False
    __X_train = []
    __Y_train = []
    __X_val = []
    __Y_val = []
    __X_test = []
    __Y_test = []

    def __init__(
            self,
            data_root: str,
            data_src: str,
            training: int,
            data_url: str,
            train_data: str,
            test_data: str,
            classes: int,
            feature_subsample: int,
            shuffle: bool,
            val_split: float,
            rand_state: int = 0,
            summary: bool = True,

            # Data type settings (for storage and data loading)
            int_dtype: type = np.int8,
            float_dtype: type = np.float16,
            output_dtype: type = np.float32,
            data_type: type = np.float16,

            # Pre-processing settings
            scaling: str = 'minmax',
         ):
        
        
        # Initialization ######################################################
        self.__data_root = data_root

        # Class-wise variables
        self.data_source = data_src
        self.training = training
        self.__rand_state = rand_state
        self.__output_dtype = output_dtype

        # Load the data #################################################
        params = {}
        params['data_url'] = data_url
        params['train_data'] = train_data
        params['test_data'] = test_data
        params['feature_subsample'] = feature_subsample
        params['classes'] = classes
        params['shuffle'] = shuffle
        params['scaling'] = scaling
        params['val_split'] = val_split
        params['data_type'] = data_type


        if not P1B2Dataset.__is_loaded:
            (X_train, Y_train), (X_val, Y_val), (X_test, Y_test) = bmk.load_data_one_hot(params, seed=rand_state)
            P1B2Dataset.__is_loaded = True           

            input_shape = X_train.shape[1:]

            P1B2Dataset.__X_train = X_train
            P1B2Dataset.__Y_train = np.argmax(Y_train, axis=1)
            P1B2Dataset.__X_val = X_val
            P1B2Dataset.__Y_val = np.argmax(Y_val, axis=1)
            P1B2Dataset.__X_test = X_test
            P1B2Dataset.__Y_test = np.argmax(Y_test, axis=1)

            P1B2Dataset.input_dim = input_shape


        if training == 1:
            self.__data = P1B2Dataset.__X_train
            self.__label = P1B2Dataset.__Y_train
            data_str = 'Training'
        elif training == 2:
            self.__data = P1B2Dataset.__X_val
            self.__label = P1B2Dataset.__Y_val 
            data_str = 'Validation'          
        else:
            self.__data = P1B2Dataset.__X_test
            self.__label = P1B2Dataset.__Y_test
            data_str = 'Test'  

        self.__out_dtype = data_type
        self.__len = len(self.__data)

        # Public attributes ###################################################
        self.input_dim = self.__data.shape
        self.num_cells = self.input_dim[0]
        self.rnaseq_dim = self.input_dim[1]

        # Dataset summary #####################################################
        if summary:
            print('=' * 80)
            print(data_str
                  + ' P1B2 Dataset Summary:')
            print('\t%i Cell Lines (feature dim: %4i).'
                  % (self.num_cells, self.rnaseq_dim))
            print('=' * 80)


    def __len__(self):
        return self.__len

    def __getitem__(self, index):

        item_data = self.__data[index]
        item_label = self.__label[index]

        item_data = np.asarray(item_data, dtype=self.__output_dtype)
        
        # Note that PyTorch requires np.int64 for classification labels
        item_label = np.int64(item_label)

        item_data = torch.as_tensor(item_data)
        item_label = torch.as_tensor(item_label)

        return item_data, item_label
