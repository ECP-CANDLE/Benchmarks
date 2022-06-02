"""
    File Name:          UnoPytorch/drug_dataframes.py
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               9/4/18
    Python Version:     3.6.6
    File Description:
        This file takes care of all the dataframes related drug features.
"""
import os
import logging
import numpy as np
import pandas as pd

from utils.data_processing.dataframe_scaling import scale_dataframe
from utils.data_processing.label_encoding import encode_label_to_int
from utils.miscellaneous.file_downloading import download_files


logger = logging.getLogger(__name__)

# Folders for raw/processed data
RAW_FOLDER = './raw/'
PROC_FOLDER = './processed/'

# All the filenames related to the drug features
ECFP_FILENAME = 'pan_drugs_dragon7_ECFP.tsv'
PFP_FILENAME = 'pan_drugs_dragon7_PFP.tsv'
DSCPTR_FILENAME = 'pan_drugs_dragon7_descriptors.tsv'
# Drug property file. Does not exist on FTP server.
DRUG_PROP_FILENAME = 'combined.panther.targets'

# Use only the following target families for classification
TGT_FAMS = ['transferase', 'oxidoreductase', 'signaling molecule',
            'nucleic acid binding', 'enzyme modulator', 'hydrolase',
            'receptor', 'transporter', 'transcription factor', 'chaperone']


def get_drug_fgpt_df(data_root: str,
                     int_dtype: type = np.int8):
    """df = get_drug_fgpt_df('./data/')

    This function loads two drug fingerprint files, join them as one
    dataframe, convert them to int_dtype and return.

    Note that if the dataframe is already stored in the processed folder,
    the function simply read from file and return after converting dtypes.

    Args:
        data_root (str): path to the data root folder.
        int_dtype (type): int dtype for storage in RAM.

    Returns:
        pd.DataFrame: processed drug fingerprint dataframe.
    """

    df_filename = 'drug_fgpt_df.pkl'
    df_path = os.path.join(data_root, PROC_FOLDER, df_filename)

    # If the dataframe already exists, load and continue ######################
    if os.path.exists(df_path):
        df = pd.read_pickle(df_path)

    # Otherwise load from raw files, process it and save ######################
    else:
        logger.debug('Processing drug fingerprint dataframe ... ')

        # Download the raw file if not exist
        download_files(filenames=[ECFP_FILENAME, PFP_FILENAME],
                       target_folder=os.path.join(data_root, RAW_FOLDER))

        ecfp_df = pd.read_csv(
            os.path.join(data_root, RAW_FOLDER, ECFP_FILENAME),
            sep='\t',
            header=None,
            index_col=0,
            skiprows=[0, ])

        pfp_df = pd.read_csv(
            os.path.join(data_root, RAW_FOLDER, PFP_FILENAME),
            sep='\t',
            header=None,
            index_col=0,
            skiprows=[0, ])

        df = pd.concat([ecfp_df, pfp_df], axis=1, join='inner')

        # Convert data type into generic python types
        df = df.astype(int)

        # save to disk for future usage
        try:
            os.makedirs(os.path.join(data_root, PROC_FOLDER))
        except FileExistsError:
            pass
        df.to_pickle(df_path)

    # Convert the dtypes for a more efficient, compact dataframe ##############
    df = df.astype(int_dtype)
    return df


def get_drug_dscptr_df(data_root: str,
                       dscptr_scaling: str,
                       dscptr_nan_thresh: float,
                       float_dtype: type = np.float32):
    """df = get_drug_dscptr_df('./data/', 'std', 0.0)

    This function loads the drug descriptor file, process it and return
    as a dataframe. The processing includes:
        * removing columns (features) and rows (drugs) that have exceeding
            ratio of NaN values comparing to nan_thresh;
        * scaling all the descriptor features accordingly;
        * convert data types for more compact structure;

    Note that if the dataframe is already stored in the processed folder,
    the function simply read from file and return after converting dtypes.

    Args:
        data_root (str): path to the data root folder.
        dscptr_scaling (str): scaling strategy for all descriptor features.
        dscptr_nan_thresh (float): threshold ratio of NaN values.
        float_dtype (float): float dtype for storage in RAM.

    Returns:
        pd.DataFrame: processed drug descriptor dataframe.
    """

    df_filename = 'drug_dscptr_df(scaling=%s, nan_thresh=%.2f).pkl' \
                  % (dscptr_scaling, dscptr_nan_thresh)
    df_path = os.path.join(data_root, PROC_FOLDER, df_filename)

    # If the dataframe already exists, load and continue ######################
    if os.path.exists(df_path):
        df = pd.read_pickle(df_path)

    # Otherwise load from raw files, process it and save ######################
    else:
        logger.debug('Processing drug descriptor dataframe ... ')

        # Download the raw file if not exist
        download_files(filenames=DSCPTR_FILENAME,
                       target_folder=os.path.join(data_root, RAW_FOLDER))

        df = pd.read_csv(
            os.path.join(data_root, RAW_FOLDER, DSCPTR_FILENAME),
            sep='\t',
            header=0,
            index_col=0,
            na_values='na')

        # Drop NaN values if the percentage of NaN exceeds nan_threshold
        # Note that columns (features) are dropped first, and then rows (drugs)
        valid_thresh = 1.0 - dscptr_nan_thresh

        df.dropna(axis=1, inplace=True, thresh=int(df.shape[0] * valid_thresh))
        df.dropna(axis=0, inplace=True, thresh=int(df.shape[1] * valid_thresh))

        # Fill the rest of NaN with column means
        df.fillna(df.mean(), inplace=True)

        # Scaling the descriptor with given scaling method
        df = scale_dataframe(df, dscptr_scaling)

        # Convert data type into generic python types
        df = df.astype(float)

        # save to disk for future usage
        try:
            os.makedirs(os.path.join(data_root, PROC_FOLDER))
        except FileExistsError:
            pass
        df.to_pickle(df_path)

    # Convert the dtypes for a more efficient, compact dataframe ##############
    df = df.astype(float_dtype)
    return df


def get_drug_feature_df(data_root: str,
                        drug_feature_usage: str,
                        dscptr_scaling: str,
                        dscptr_nan_thresh: float,
                        int_dtype: type = np.int8,
                        float_dtype: type = np.float32):
    """df = get_drug_feature_df('./data/', 'both', 'std', 0.0)

    This function utilizes get_drug_fgpt_df and get_drug_dscptr_df. If the
    feature usage is 'both', it will loads fingerprint and descriptors,
    join them and return. Otherwise, if feature usage is set to
    'fingerprint' or 'descriptor', the function returns the corresponding
    dataframe.

    Args:
        data_root (str): path to the data root folder.
        drug_feature_usage (str): feature usage indicator. Choose between
            'both', 'fingerprint', and 'descriptor'.
        dscptr_scaling (str): scaling strategy for all descriptor features.
        dscptr_nan_thresh (float): threshold ratio of NaN values.
        int_dtype (type): int dtype for storage in RAM.
        float_dtype (float): int dtype for storage in RAM.

    Returns:
        pd.DataFrame: processed drug feature dataframe.
    """

    # Return the corresponding drug feature dataframe
    if drug_feature_usage == 'both':
        return pd.concat(
            [get_drug_fgpt_df(data_root=data_root,
                              int_dtype=int_dtype),
             get_drug_dscptr_df(data_root=data_root,
                                dscptr_scaling=dscptr_scaling,
                                dscptr_nan_thresh=dscptr_nan_thresh,
                                float_dtype=float_dtype)],
            axis=1, join='inner')
    elif drug_feature_usage == 'fingerprint':
        return get_drug_fgpt_df(data_root=data_root,
                                int_dtype=int_dtype)
    elif drug_feature_usage == 'descriptor':
        return get_drug_dscptr_df(data_root=data_root,
                                  dscptr_scaling=dscptr_scaling,
                                  dscptr_nan_thresh=dscptr_nan_thresh,
                                  float_dtype=float_dtype)
    else:
        logger.error('Drug feature must be one of \'fingerprint\', '
                     '\'descriptor\', or \'both\'.', exc_info=True)
        raise ValueError('Undefined drug feature %s.' % drug_feature_usage)


def get_drug_prop_df(data_root: str):
    """df = get_drug_prop_df('./data/')

    This function loads the drug property file and returns a dataframe with
    only weighted QED and target families as columns (['QED', 'TARGET']).

    Note that if the dataframe is already stored in the processed folder,
    the function simply read from file and return after converting dtypes.

    Args:
        data_root (str): path to the data root folder.

    Returns:
        pd.DataFrame: drug property dataframe with target families and QED.
    """

    df_filename = 'drug_prop_df.pkl'
    df_path = os.path.join(data_root, PROC_FOLDER, df_filename)

    # If the dataframe already exists, load and continue ######################
    if os.path.exists(df_path):
        df = pd.read_pickle(df_path)

    # Otherwise load from raw files, process it and save ######################
    else:
        logger.debug('Processing drug targets dataframe ... ')

        raw_file_path = os.path.join(data_root, RAW_FOLDER, DRUG_PROP_FILENAME)

        # Download the raw file if not exist
        download_files(filenames=DRUG_PROP_FILENAME,
                       target_folder=os.path.join(data_root, RAW_FOLDER))

        df = pd.read_csv(
            raw_file_path,
            sep='\t',
            header=0,
            index_col=0,
            usecols=['anl_cpd_id', 'qed_weighted', 'target_families', ])

        # Change index name for consistency across the whole program
        df.index.names = ['DRUG_ID']
        df.columns = ['QED', 'TARGET', ]

        # Convert data type into generic python types
        df[['QED']] = df[['QED']].astype(float)
        df[['TARGET']] = df[['TARGET']].astype(str)

        # save to disk for future usage
        try:
            os.makedirs(os.path.join(data_root, PROC_FOLDER))
        except FileExistsError:
            pass
        df.to_pickle(df_path)

    return df


def get_drug_target_df(data_root: str,
                       int_dtype: type = np.int8):
    """df = get_drug_target_df('./data/')

    This function the drug property dataframe, process it and return the
    drug target families dataframe. The processing includes:
        * removing all columns but 'TARGET';
        * drop drugs/rows that are not in the TGT_FAMS list;
        * encode target families into integer labels;
        * convert data types for more compact structure;

    Args:
        data_root (str): path to the data root folder.
        int_dtype (type): int dtype for storage in RAM.

    Returns:
        pd.DataFrame: drug target families dataframe.
    """

    df = get_drug_prop_df(data_root=data_root)[['TARGET']]

    # Only take the rows with specific target families for classification
    df = df[df['TARGET'].isin(TGT_FAMS)][['TARGET']]

    # Encode str formatted target families into integers
    df['TARGET'] = encode_label_to_int(data_root=data_root,
                                       dict_name='drug_target_dict.txt',
                                       labels=df['TARGET'])

    # Convert the dtypes for a more efficient, compact dataframe
    # Note that it is safe to use int8 here for there are only 10 classes
    return df.astype(int_dtype)


def get_drug_qed_df(data_root: str,
                    qed_scaling: str,
                    float_dtype: type = np.float32):
    """df = get_drug_qed_df('./data/', 'none')


    This function the drug property dataframe, process it and return the
    drug weighted QED dataframe. The processing includes:
        * removing all columns but 'QED';
        * drop drugs/rows that have NaN as weighted QED;
        * scaling the QED accordingly;
        * convert data types for more compact structure;

    Args:
        data_root (str): path to the data root folder.
        qed_scaling (str): scaling strategy for weighted QED.
        float_dtype (float): float dtype for storage in RAM.

    Returns:
        pd.DataFrame: drug weighted QED dataframe.
    """

    df = get_drug_prop_df(data_root=data_root)[['QED']]

    # Drop all the NaN values before scaling
    df.dropna(axis=0, inplace=True)

    # Note that weighted QED is by default already in the range of [0, 1]
    # Scaling the weighted QED with given scaling method
    df = scale_dataframe(df, qed_scaling)

    # Convert the dtypes for a more efficient, compact dataframe
    return df.astype(float_dtype)


if __name__ == '__main__':

    logging.basicConfig(level=logging.DEBUG)

    print('=' * 80 + '\nDrug feature dataframe head:')
    print(get_drug_feature_df(data_root='../../data/',
                              drug_feature_usage='both',
                              dscptr_scaling='std',
                              dscptr_nan_thresh=0.).head())

    print('=' * 80 + '\nDrug target families dataframe head:')
    print(get_drug_target_df(data_root='../../data/').head())

    print('=' * 80 + '\nDrug target families dataframe head:')
    print(get_drug_qed_df(data_root='../../data/', qed_scaling='none').head())
