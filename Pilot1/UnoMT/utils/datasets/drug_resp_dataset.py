"""
    File Name:          UnoPytorch/drug_resp_dataset.py
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               8/13/18
    Python Version:     3.6.6
    File Description:
        This file implements the dataset for drug response.
"""

import logging
import numpy as np
import torch.utils.data as data
from sklearn.model_selection import train_test_split

from utils.data_processing.cell_line_dataframes import get_rna_seq_df, \
    get_cl_meta_df
from utils.data_processing.drug_dataframes import get_drug_feature_df
from utils.data_processing.label_encoding import get_label_dict
from utils.data_processing.response_dataframes import get_drug_resp_df, \
    get_drug_anlys_df

logger = logging.getLogger(__name__)


class DrugRespDataset(data.Dataset):
    """Dataset class for drug response learning.

    This class implements a PyTorch Dataset class made for drug response
    learning. Using enumerate() or any other methods that utilize
    __getitem__() to access the data.

    Each data item is made of a tuple of (feature, target), where feature is
    a list including drug and cell line information along with the log
    concentration, and target is the growth.

    Note that all items in feature and the target are in python float type.

    Attributes:
        training (bool): indicator of training/validation dataset.
        drugs (list): list of all the drugs in the dataset.
        cells (list): list of all the cells in the dataset.
        data_source (str): source of the data being used.
        num_records (int): number of drug response records.
        drug_feature_dim (int): dimensionality of drug feature.
        rnaseq_dim (int): dimensionality of RNA sequence.
    """

    def __init__(
            self,
            data_root: str,
            data_src: str,
            training: bool,
            rand_state: int = 0,
            summary: bool = True,

            # Data type settings (for storage and data loading)
            int_dtype: type = np.int8,
            float_dtype: type = np.float16,
            output_dtype: type = np.float32,

            # Pre-processing settings
            grth_scaling: str = 'none',
            dscptr_scaling: str = 'std',
            rnaseq_scaling: str = 'std',
            dscptr_nan_threshold: float = 0.0,

            # Partitioning (train/validation) and data usage settings
            rnaseq_feature_usage: str = 'source_scale',
            drug_feature_usage: str = 'both',
            validation_ratio: float = 0.2,
            disjoint_drugs: bool = True,
            disjoint_cells: bool = True, ):
        """dataset = DrugRespDataset('./data/', 'NCI60', True)

        Construct a new drug response dataset based on the parameters
        provided. The process includes:
            * Downloading source data files;
            * Pre-processing (scaling, trimming, etc.);
            * Public attributes and other preparations.

        Args:
            data_root (str): path to data root folder.
            data_src (str): data source for drug response, must be one of
                'NCI60', 'CTRP', 'GDSC', 'CCLE', 'gCSI', and 'all'.
            training (bool): indicator for training.
            rand_state (int): random seed used for training/validation split
                and other processes that requires randomness.
            summary (bool): set True for printing dataset summary.

            int_dtype (type): integer dtype for data storage in RAM.
            float_dtype (type): float dtype for data storage in RAM.
            output_dtype (type): output dtype for neural network.

            grth_scaling (str): scaling method for drug response growth.
                Choose between 'none', 'std', and 'minmax'.
            dscptr_scaling (str): scaling method for drug descriptor.
                Choose between 'none', 'std', and 'minmax'.
            rnaseq_scaling (str): scaling method for RNA sequence (LINCS1K).
                Choose between 'none', 'std', and 'minmax'.
            dscptr_nan_threshold (float): NaN threshold for drug descriptor.
                If a column/feature or row/drug contains exceeding amount of
                NaN comparing to the threshold, the feature/drug will be
                dropped.

            rnaseq_feature_usage (str): RNA sequence usage. Choose between
                'combat', which is batch-effect-removed version of RNA
                sequence, or 'source_scale'.
            drug_feature_usage (str): drug feature usage. Choose between
                'fingerprint', 'descriptor', or 'both'.
            validation_ratio (float): portion of validation data out of all
                data samples. Note that this is not strictly the portion
                size. During the split, we will pick a percentage of
                drugs/cells and take the combination. The calculation will
                make sure that the expected validation size is accurate,
                but not strictly the case for a single random seed. Please
                refer to __split_drug_resp() for more details.
            disjoint_drugs (bool): indicator for disjoint drugs between
                training and validation dataset.
            disjoint_cells: indicator for disjoint cell lines between
                training and validation dataset.
        """

        # Initialization ######################################################
        self.__data_root = data_root

        # Class-wise variables
        self.data_source = data_src
        self.training = training
        self.__rand_state = rand_state
        self.__output_dtype = output_dtype

        # Feature scaling
        if grth_scaling is None or grth_scaling == '':
            grth_scaling = 'none'
        grth_scaling = grth_scaling.lower()
        if dscptr_scaling is None or dscptr_scaling == '':
            dscptr_scaling = 'none'
        dscptr_scaling = dscptr_scaling
        if rnaseq_scaling is None or rnaseq_scaling == '':
            rnaseq_scaling = 'none'
        rnaseq_scaling = rnaseq_scaling

        self.__validation_ratio = validation_ratio
        self.__disjoint_drugs = disjoint_drugs
        self.__disjoint_cells = disjoint_cells

        # Load all dataframes #################################################
        self.__drug_resp_df = get_drug_resp_df(
            data_root=data_root,
            grth_scaling=grth_scaling,
            int_dtype=int_dtype,
            float_dtype=float_dtype)

        self.__drug_feature_df = get_drug_feature_df(
            data_root=data_root,
            drug_feature_usage=drug_feature_usage,
            dscptr_scaling=dscptr_scaling,
            dscptr_nan_thresh=dscptr_nan_threshold,
            int_dtype=int_dtype,
            float_dtype=float_dtype)

        self.__rnaseq_df = get_rna_seq_df(
            data_root=data_root,
            rnaseq_feature_usage=rnaseq_feature_usage,
            rnaseq_scaling=rnaseq_scaling,
            float_dtype=float_dtype)

        # Train/validation split ##############################################
        self.__split_drug_resp()

        # Public attributes ###################################################
        self.drugs = self.__drug_resp_df['DRUG_ID'].unique().tolist()
        self.cells = self.__drug_resp_df['CELLNAME'].unique().tolist()
        self.num_records = len(self.__drug_resp_df)
        self.drug_feature_dim = self.__drug_feature_df.shape[1]
        self.rnaseq_dim = self.__rnaseq_df.shape[1]

        # Converting dataframes to arrays and dict for rapid access ###########
        self.__drug_resp_array = self.__drug_resp_df.values
        # The following conversion will upcast dtypes
        self.__drug_feature_dict = {idx: row.values for idx, row in
                                    self.__drug_feature_df.iterrows()}
        self.__rnaseq_dict = {idx: row.values for idx, row in
                              self.__rnaseq_df.iterrows()}

        # Dataframes are not needed any more
        self.__drug_resp_df = None
        self.__drug_feature_df = None
        self.__rnaseq_df = None

        # Dataset summary #####################################################
        if summary:
            print('=' * 80)
            print(('Training' if self.training else 'Validation')
                  + ' Drug Response Dataset Summary (Data Source: %6s):'
                  % self.data_source)
            print('\t%i Drug Response Records .' % len(self.__drug_resp_array))
            print('\t%i Unique Drugs (feature dim: %4i).'
                  % (len(self.drugs), self.drug_feature_dim))
            print('\t%i Unique Cell Lines (feature dim: %4i).'
                  % (len(self.cells), self.rnaseq_dim))
            print('=' * 80)

    def __len__(self):
        return self.num_records

    def __getitem__(self, index):
        """rnaseq, drug_feature, concentration, growth = dataset[0]

        This function fetches a single sample of drug response data along
        with the corresponding drug features and RNA sequence.

        Note that all the returned values are in ndarray format with the
        type specified during dataset initialization.

        Args:
            index (int): index for drug response data.

        Returns:
            tuple: a tuple of np.ndarray, with RNA sequence data,
                drug features, concentration, and growth.
        """

        # Note that this chunk of code does not work with pytorch 4.1 for
        # multiprocessing reasons. Chances are that during the run, one of
        # the workers might hang and prevents the training from moving on

        # Note that even with locks for multiprocessing lib, the code will
        # get stuck at some point if all CPU cores are used.

        drug_resp = self.__drug_resp_array[index]
        drug_feature = self.__drug_feature_dict[drug_resp[1]]
        rnaseq = self.__rnaseq_dict[drug_resp[2]]

        drug_feature = drug_feature.astype(self.__output_dtype)
        rnaseq = rnaseq.astype(self.__output_dtype)
        concentration = np.array([drug_resp[3]], dtype=self.__output_dtype)
        growth = np.array([drug_resp[4]], dtype=self.__output_dtype)

        return rnaseq, drug_feature, concentration, growth

    def __trim_dataframes(self):
        """self.__trim_dataframes(trim_data_source=True)

        This function trims three dataframes to make sure that drug response
        dataframe, RNA sequence dataframe, and drug feature dataframe are
        sharing the same list of cell lines and drugs.

        Returns:
            None
        """

        # Encode the data source and take the data from target source only
        # Note that source could be 'NCI60', 'GDSC', etc. and 'all'
        if self.data_source.lower() != 'all':

            logger.debug('Specifying data source %s ... ' % self.data_source)

            data_src_dict = get_label_dict(data_root=self.__data_root,
                                           dict_name='data_src_dict.txt')
            encoded_data_src = data_src_dict[self.data_source]

            # Reduce/trim the drug response dataframe
            self.__drug_resp_df = self.__drug_resp_df.loc[
                self.__drug_resp_df['SOURCE'] == encoded_data_src]

        # Make sure that all three dataframes share the same drugs/cells
        logger.debug('Trimming dataframes on common cell lines and drugs ... ')

        cell_set = set(self.__drug_resp_df['CELLNAME'].unique()) \
            & set(self.__rnaseq_df.index.values)
        drug_set = set(self.__drug_resp_df['DRUG_ID'].unique()) \
            & set(self.__drug_feature_df.index.values)

        self.__drug_resp_df = self.__drug_resp_df.loc[
            (self.__drug_resp_df['CELLNAME'].isin(cell_set))
            & (self.__drug_resp_df['DRUG_ID'].isin(drug_set))]

        self.__rnaseq_df = self.__rnaseq_df[
            self.__rnaseq_df.index.isin(cell_set)]
        self.__drug_feature_df = self.__drug_feature_df[
            self.__drug_feature_df.index.isin(drug_set)]

        logger.debug('There are %i drugs and %i cell lines, with %i response '
                     'records after trimming.'
                     % (len(drug_set), len(cell_set),
                        len(self.__drug_resp_df)))
        return

    def __split_drug_resp(self):
        """self.__split_drug_resp()

        This function split training and validation drug response data based
        on the splitting specifications (disjoint drugs and/or disjoint cells).

        Upon the call, the function summarize all the drugs and cells. If
        disjoint (drugs/cells) is set to True, then it will split the list
        (of drugs/cells) into training/validation (drugs/cells).

        Otherwise, if disjoint (drugs/cells) is set to False, we make sure
        that the training/validation set contains the same (drugs/cells).

        Then it trims all three dataframes to make sure all the data in RAM is
        relevant for training/validation

        Note that the validation size is not guaranteed during splitting.
        What the function really splits by the ratio is the list of
        drugs/cell lines. Also, if both drugs and cell lines are marked
        disjoint, the function will split drug and cell lists with ratio of
        (validation_size ** 0.7).

        Warnings will be raise if the validation ratio is off too much.

        Returns:
            None
        """

        # Trim dataframes based on data source and common drugs/cells
        # Now drug response dataframe contains training + validation
        # data samples from the same data source, like 'NCI60'
        self.__trim_dataframes()

        # Get lists of all drugs & cells corresponding from data source
        cell_list = self.__drug_resp_df['CELLNAME'].unique().tolist()
        drug_list = self.__drug_resp_df['DRUG_ID'].unique().tolist()

        # Create an array to store all drugs' analysis results
        drug_anlys_dict = {idx: row.values for idx, row in
                           get_drug_anlys_df(self.__data_root).iterrows()}
        drug_anlys_array = np.array([drug_anlys_dict[d] for d in drug_list])

        # Create a list to store all cell lines types
        cell_type_dict = {idx: row.values for idx, row in
                          get_cl_meta_df(self.__data_root)
                          [['type']].iterrows()}
        cell_type_list = [cell_type_dict[c] for c in cell_list]

        # Change validation size when both features are disjoint in splitting
        # Note that theoretically should use validation_ratio ** 0.5,
        # but 0.7 simply works better in most cases.
        if self.__disjoint_cells and self.__disjoint_drugs:
            adjusted_val_ratio = self.__validation_ratio ** 0.7
        else:
            adjusted_val_ratio = self.__validation_ratio

        split_kwargs = {
            'test_size': adjusted_val_ratio,
            'random_state': self.__rand_state,
            'shuffle': True, }

        # Try to split the cells stratified on type list
        try:
            training_cell_list, validation_cell_list = \
                train_test_split(cell_list, **split_kwargs,
                                 stratify=cell_type_list)
        except ValueError:
            logger.warning('Failed to split %s cells in stratified '
                           'way. Splitting randomly ...' % self.data_source)
            training_cell_list, validation_cell_list = \
                train_test_split(cell_list, **split_kwargs)

        # Try to split the drugs stratified on the drug analysis results
        try:
            training_drug_list, validation_drug_list = \
                train_test_split(drug_list, **split_kwargs,
                                 stratify=drug_anlys_array)
        except ValueError:
            logger.warning('Failed to split %s drugs stratified on growth '
                           'and correlation. Splitting solely on avg growth'
                           ' ...' % self.data_source)

            try:
                training_drug_list, validation_drug_list = \
                    train_test_split(drug_list, **split_kwargs,
                                     stratify=drug_anlys_array[:, 0])
            except ValueError:
                logger.warning('Failed to split %s drugs on avg growth. '
                               'Splitting solely on avg correlation ...'
                               % self.data_source)

                try:
                    training_drug_list, validation_drug_list = \
                        train_test_split(drug_list, **split_kwargs,
                                         stratify=drug_anlys_array[:, 1])
                except ValueError:
                    logger.warning('Failed to split %s drugs on avg '
                                   'correlation. Splitting randomly ...'
                                   % self.data_source)
                    training_drug_list, validation_drug_list = \
                        train_test_split(drug_list, **split_kwargs)

        # Split data based on disjoint cell/drug strategy
        if self.__disjoint_cells and self.__disjoint_drugs:

            training_drug_resp_df = self.__drug_resp_df.loc[
                (self.__drug_resp_df['CELLNAME'].isin(training_cell_list))
                & (self.__drug_resp_df['DRUG_ID'].isin(training_drug_list))]

            validation_drug_resp_df = self.__drug_resp_df.loc[
                (self.__drug_resp_df['CELLNAME'].isin(validation_cell_list))
                & (self.__drug_resp_df['DRUG_ID'].isin(validation_drug_list))]

        elif self.__disjoint_cells and (not self.__disjoint_drugs):

            training_drug_resp_df = self.__drug_resp_df.loc[
                self.__drug_resp_df['CELLNAME'].isin(training_cell_list)]

            validation_drug_resp_df = self.__drug_resp_df.loc[
                self.__drug_resp_df['CELLNAME'].isin(validation_cell_list)]

        elif (not self.__disjoint_cells) and self.__disjoint_drugs:

            training_drug_resp_df = self.__drug_resp_df.loc[
                self.__drug_resp_df['DRUG_ID'].isin(training_drug_list)]

            validation_drug_resp_df = self.__drug_resp_df.loc[
                self.__drug_resp_df['DRUG_ID'].isin(validation_drug_list)]

        else:
            training_drug_resp_df, validation_drug_resp_df = \
                train_test_split(self.__drug_resp_df,
                                 test_size=self.__validation_ratio,
                                 random_state=self.__rand_state,
                                 shuffle=False)

        # Make sure that if not disjoint, the training/validation set should
        #  share the same drugs/cells
        if not self.__disjoint_cells:
            # Make sure that cell lines are common
            common_cells = set(training_drug_resp_df['CELLNAME'].unique()) & \
                set(validation_drug_resp_df['CELLNAME'].unique())

            training_drug_resp_df = training_drug_resp_df.loc[
                training_drug_resp_df['CELLNAME'].isin(common_cells)]
            validation_drug_resp_df = validation_drug_resp_df.loc[
                validation_drug_resp_df['CELLNAME'].isin(common_cells)]

        if not self.__disjoint_drugs:
            # Make sure that drugs are common
            common_drugs = set(training_drug_resp_df['DRUG_ID'].unique()) & \
                set(validation_drug_resp_df['DRUG_ID'].unique())

            training_drug_resp_df = training_drug_resp_df.loc[
                training_drug_resp_df['DRUG_ID'].isin(common_drugs)]
            validation_drug_resp_df = validation_drug_resp_df.loc[
                validation_drug_resp_df['DRUG_ID'].isin(common_drugs)]

        # Check if the validation ratio is in a reasonable range
        validation_ratio = len(validation_drug_resp_df) \
            / (len(training_drug_resp_df) + len(validation_drug_resp_df))
        if (validation_ratio < self.__validation_ratio * 0.8) \
                or (validation_ratio > self.__validation_ratio * 1.2):
            logger.warning('Bad validation ratio: %.3f' %
                           validation_ratio)

        # Keep only training_drug_resp_df or validation_drug_resp_df
        self.__drug_resp_df = training_drug_resp_df if self.training \
            else validation_drug_resp_df


# Test segment for drug response dataset
if __name__ == '__main__':

    logging.basicConfig(level=logging.DEBUG)

    for src in ['NCI60', 'CTRP', 'GDSC', 'CCLE', 'gCSI']:

        kwarg = {
            'data_root': '../../data/',
            'summary': False,
            'rand_state': 0, }

        trn_set = DrugRespDataset(
            data_src=src,
            training=True,
            **kwarg)

        val_set = DrugRespDataset(
            data_src=src,
            training=False,
            **kwarg)

        print('There are %i drugs and %i cell lines in %s.'
              % ((len(trn_set.drugs) + len(val_set.drugs)),
                 (len(trn_set.cells) + len(val_set.cells)), src))
