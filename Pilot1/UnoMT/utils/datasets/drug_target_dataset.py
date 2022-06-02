"""
    File Name:          UnoPytorch/drug_target_dataset.py
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               9/4/18
    Python Version:     3.6.6
    File Description:

"""

import logging
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from sklearn.model_selection import train_test_split

from utils.data_processing.drug_dataframes import get_drug_target_df, \
    get_drug_feature_df

logger = logging.getLogger(__name__)


class DrugTargetDataset(data.Dataset):

    def __init__(
            self,
            data_root: str,
            training: bool,
            rand_state: int = 0,
            summary: bool = True,

            # Data type settings (for storage and data loading)
            int_dtype: type = np.int8,
            float_dtype: type = np.float16,
            output_dtype: type = np.float32,

            # Pre-processing settings
            dscptr_scaling: str = 'std',
            dscptr_nan_threshold: float = 0.0,

            # Partitioning (train/validation) and data usage settings
            drug_feature_usage: str = 'both',
            validation_ratio: float = 0.2, ):
        # Initialization ######################################################
        # Class-wise variables
        self.training = training
        self.__rand_state = rand_state
        self.__output_dtype = output_dtype

        self.__validation_ratio = validation_ratio

        # Load all dataframes #################################################
        self.__drug_feature_df = get_drug_feature_df(
            data_root=data_root,
            drug_feature_usage=drug_feature_usage,
            dscptr_scaling=dscptr_scaling,
            dscptr_nan_thresh=dscptr_nan_threshold,
            int_dtype=int_dtype,
            float_dtype=float_dtype)

        self.__drug_target_df = get_drug_target_df(
            data_root=data_root,
            int_dtype=int_dtype)

        # Put all the drug feature in one column as a list with dtype
        self.__drug_feature_df['feature'] = \
            list(map(float_dtype, self.__drug_feature_df.values.tolist()))

        # Join the drug features with drug target family labels
        self.__drug_target_df = pd.concat(
            [self.__drug_target_df, self.__drug_feature_df[['feature']]],
            axis=1, join='inner')

        # Train/validation split ##############################################
        self.__split_drug_resp()

        # Public attributes ###################################################
        self.drugs = self.__drug_target_df.index.tolist()
        self.num_drugs = len(self.drugs)
        self.drug_feature_dim = self.__drug_feature_df.shape[1]

        assert self.num_drugs == len(self.__drug_target_df)

        # Converting dataframes to arrays and dict for rapid access ###########
        self.__drug_target_array = self.__drug_target_df.values

        # Clear the dataframes ################################################
        self.__drug_feature_df = None
        self.__drug_target_df = None

        # Dataset summary #####################################################
        if summary:
            print('=' * 80)
            print(('Training' if self.training else 'Validation') + ' Drug Target Family Classification Dataset Summary:')
            print('\t%i Unique Drugs (feature dim: %4i).'
                  % (self.num_drugs, self.drug_feature_dim))
            print('=' * 80)

    def __len__(self):
        return self.num_drugs

    def __getitem__(self, index):
        drug_target_data = self.__drug_target_array[index]

        drug_feature = np.asarray(drug_target_data[1],
                                  dtype=self.__output_dtype)
        target = np.int64(drug_target_data[0])

        return drug_feature, target

    def __split_drug_resp(self):

        training_df, validation_df = \
            train_test_split(self.__drug_target_df,
                             test_size=self.__validation_ratio,
                             random_state=self.__rand_state,
                             shuffle=True)

        self.__drug_target_df = training_df if self.training else validation_df


if __name__ == '__main__':

    logging.basicConfig(level=logging.DEBUG)

    # Test DrugRespDataset class
    dataloader = torch.utils.data.DataLoader(
        DrugTargetDataset(data_root='../../data/',
                          training=True),
        batch_size=512, shuffle=False)

    tmp = dataloader.dataset[0]
    print(tmp)
