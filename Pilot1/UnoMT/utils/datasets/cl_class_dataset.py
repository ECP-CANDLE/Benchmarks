"""
    File Name:          UnoPytorch/cl_class_dataset.py
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               8/13/18
    Python Version:     3.6.6
    File Description:
        This file implements the dataset for cell line classification.
"""

import logging

import torch
import numpy as np
import pandas as pd
import torch.utils.data as data
from sklearn.model_selection import train_test_split

from utils.data_processing.cell_line_dataframes import get_rna_seq_df, \
    get_cl_meta_df
from utils.data_processing.label_encoding import encode_int_to_onehot, \
    get_label_dict

logger = logging.getLogger(__name__)


class CLClassDataset(data.Dataset):
    """Dataset class for cell line classification

    This class implements a PyTorch Dataset class made for cell line
    classification. Using enumerate() or any other methods that utilize
    __getitem__() to access the data.

    Each data item is made of a tuple of
        (RNA_sequence, conditions, site, type, category)
    where conditions is a list of [data_source, cell_description].

    Note that all categorical labels are numeric, and the encoding
    dictionary can be found in the processed folder.

    Attributes:
        training (bool): indicator of training/validation dataset
        cells (list): list of all the cells in the dataset
        num_cells (int): number of cell lines in the dataset
        rnaseq_dim (int): dimensionality of RNA sequence
    """

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
            rnaseq_scaling: str = 'std',

            # Partitioning (train/validation) and data usage settings
            rnaseq_feature_usage: str = 'source_scale',
            validation_ratio: float = 0.2, ):
        """dataset = CLClassDataset('./data/', True)

        Construct a RNA sequence dataset based on the parameters provided.
        The process includes:
            * Downloading source data files;
            * Pre-processing (scaling);
            * Public attributes and other preparations.

        Args:
            data_root (str): path to data root folder.
            training (bool): indicator for training.
            rand_state (int): random seed used for training/validation split
                and other processes that requires randomness.
            summary (bool): set True for printing dataset summary.

            int_dtype (type): integer dtype for data storage in RAM.
            float_dtype (type): float dtype for data storage in RAM.
            output_dtype (type): output dtype for neural network.

            rnaseq_scaling (str): scaling method for RNA squence. Choose
                between 'none', 'std', and 'minmax'.
            rnaseq_feature_usage: RNA sequence data usage. Choose between
                'source_scale' and 'combat'.
            validation_ratio (float): portion of validation data out of all
                data samples.
        """

        # Initialization ######################################################
        self.__data_root = data_root

        # Class-wise variables
        self.training = training
        self.__rand_state = rand_state
        self.__output_dtype = output_dtype

        # Feature scaling
        if rnaseq_scaling is None or rnaseq_scaling == '':
            rnaseq_scaling = 'none'
        self.__rnaseq_scaling = rnaseq_scaling.lower()

        self.__rnaseq_feature_usage = rnaseq_feature_usage
        self.__validation_ratio = validation_ratio

        # Load all dataframes #################################################
        self.__rnaseq_df = get_rna_seq_df(
            data_root=data_root,
            rnaseq_feature_usage=rnaseq_feature_usage,
            rnaseq_scaling=rnaseq_scaling,
            float_dtype=float_dtype)

        self.__cl_meta_df = get_cl_meta_df(
            data_root=data_root,
            int_dtype=int_dtype)

        # Put all the sequence in one column as list and specify dtype
        self.__rnaseq_df['seq'] = \
            list(map(float_dtype, self.__rnaseq_df.values.tolist()))

        # Join the RNA sequence data with meta data. cl_df will have columns:
        # ['data_src', 'site', 'type', 'category', 'seq']
        self.__cl_df = pd.concat([self.__cl_meta_df,
                                  self.__rnaseq_df[['seq']]],
                                 axis=1, join='inner')

        # Encode data source from int into one-hot encoding
        num_data_src = len(get_label_dict(data_root, 'data_src_dict.txt'))
        enc_data_src = encode_int_to_onehot(self.__cl_df['data_src'].tolist(),
                                            num_classes=num_data_src)
        self.__cl_df['data_src'] = list(map(int_dtype, enc_data_src))

        # Train/validation split ##############################################
        self.__split_drug_resp()

        # Converting dataframes to arrays for rapid access ####################
        self.__cl_array = self.__cl_df.values

        # Public attributes ###################################################
        self.cells = self.__cl_df.index.tolist()
        self.num_cells = self.__cl_df.shape[0]
        self.rnaseq_dim = len(self.__cl_df.iloc[0]['seq'])

        # Clear the dataframes ################################################
        self.__rnaseq_df = None
        self.__cl_meta_df = None
        self.__cl_df = None

        # Dataset summary #####################################################
        if summary:
            print('=' * 80)
            print(('Training' if self.training else 'Validation')
                  + ' RNA Sequence Dataset Summary:')
            print('\t%i Unique Cell Lines (feature dim: %4i).'
                  % (self.num_cells, self.rnaseq_dim))
            print('=' * 80)

    def __len__(self):
        """length = len(cl_class_dataset)

        Get the length of dataset, which is the number of cell lines.

        Returns:
            int: the length of dataset.
        """
        return self.num_cells

    def __getitem__(self, index):
        """rnaseq, data_src, site, type, category = cl_class_dataset[0]

        Args:
            index (int): index for target data slice.

        Returns:
            tuple: a tuple containing the following five elements:
                * RNA sequence data (np.ndarray of float);
                * one-hot-encoded data source (np.ndarray of float);
                * encoded cell line site (int);
                * encoded cell line type (int);
                * encoded cell line category (int)
        """

        cl_data = self.__cl_array[index]

        rnaseq = np.asarray(cl_data[4], dtype=self.__output_dtype)
        data_src = np.array(cl_data[0], dtype=self.__output_dtype)

        # Note that PyTorch requires np.int64 for classification labels
        cl_site = np.int64(cl_data[1])
        cl_type = np.int64(cl_data[2])
        cl_category = np.int64(cl_data[3])

        return rnaseq, data_src, cl_site, cl_type, cl_category

    def __split_drug_resp(self):
        """self.__split_drug_resp()

        Split training and validation dataframe for cell lines, stratified
        on tumor type. Note that after the split, our dataframe will only
        contain training/validation data based on training indicator.

        Returns:
            None
        """
        split_kwargs = {
            'test_size': self.__validation_ratio,
            'random_state': self.__rand_state,
            'shuffle': True, }

        try:
            training_cl_df, validation_cl_df = \
                train_test_split(self.__cl_df, **split_kwargs,
                                 stratify=self.__cl_df['type'].tolist())
        except ValueError:
            logger.warning('Failed to split cell lines in stratified way. '
                           'Splitting randomly ...')
            training_cl_df, validation_cl_df = \
                train_test_split(self.__cl_df, **split_kwargs)

        self.__cl_df = training_cl_df if self.training else validation_cl_df


# Test segment for cell line classification dataset
if __name__ == '__main__':

    logging.basicConfig(level=logging.DEBUG)

    # Test DrugRespDataset class
    dataloader = torch.utils.data.DataLoader(
        CLClassDataset(data_root='../../data/',
                       training=False),
        batch_size=512, shuffle=False)

    tmp = dataloader.dataset[0]
