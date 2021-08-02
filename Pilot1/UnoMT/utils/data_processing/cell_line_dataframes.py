"""
    File Name:          UnoPytorch/cell_line_dataframes.py
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               9/4/18
    Python Version:     3.6.6
    File Description:
        This file takes care of all the dataframes related cell lines.
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

# All the filenames related to cell lines
CL_METADATA_FILENAME = 'combined_cl_metadata'
RNASEQ_SOURCE_SCALE_FILENAME = 'combined_rnaseq_data_lincs1000_source_scale'
RNASEQ_COMBAT_FILENAME = 'combined_rnaseq_data_lincs1000_combat'


def get_rna_seq_df(data_root: str,
                   rnaseq_feature_usage: str,
                   rnaseq_scaling: str,
                   float_dtype: type = np.float32):
    """df = get_rna_seq_df('./data/', 'source_scale', 'std')

    This function loads the RNA sequence file, process it and return
    as a dataframe. The processing includes:
        * remove the '-' in cell line names;
        * remove duplicate indices;
        * scaling all the sequence features accordingly;
        * convert data types for more compact structure;

    Note that if the dataframe is already stored in the processed folder,
    the function simply read from file and return after converting dtypes.

    Args:
        data_root (str): path to the data root folder.
        rnaseq_feature_usage (str): feature usage indicator, Choose between
            'source_scale' and 'combat'.
        rnaseq_scaling (str): scaling strategy for RNA sequence.
        float_dtype (float): int dtype for storage in RAM.

    Returns:
        pd.DataFrame: processed RNA sequence dataframe.
    """

    df_filename = 'rnaseq_df(%s, scaling=%s).pkl' \
                  % (rnaseq_feature_usage, rnaseq_scaling)
    df_path = os.path.join(data_root, PROC_FOLDER, df_filename)

    # If the dataframe already exists, load and continue ######################
    if os.path.exists(df_path):
        df = pd.read_pickle(df_path)

    # Otherwise load from raw files, process it and save ######################
    else:
        logger.debug('Processing RNA sequence dataframe ... ')

        if rnaseq_feature_usage == 'source_scale':
            raw_data_filename = RNASEQ_SOURCE_SCALE_FILENAME
        elif rnaseq_feature_usage == 'combat':
            raw_data_filename = RNASEQ_COMBAT_FILENAME
        else:
            logger.error('Unknown RNA feature %s.' % rnaseq_feature_usage,
                         exc_info=True)
            raise ValueError('RNA feature usage must be one of '
                             '\'source_scale\' or \'combat\'.')

        # Download the raw file if not exist
        download_files(filenames=raw_data_filename,
                       target_folder=os.path.join(data_root, RAW_FOLDER))

        df = pd.read_csv(
            os.path.join(data_root, RAW_FOLDER, raw_data_filename),
            sep='\t',
            header=0,
            index_col=0)

        # Delete '-', which could be inconsistent between seq and meta
        df.index = df.index.str.replace('-', '')

        # Note that after this name changing, some rows will have the same
        # name like 'GDSC.TT' and 'GDSC.T-T', but they are actually the same
        # Drop the duplicates for consistency
        print(df.shape)
        df = df[~df.index.duplicated(keep='first')]
        print(df.shape)

        # Scaling the descriptor with given scaling method
        df = scale_dataframe(df, rnaseq_scaling)

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


def get_cl_meta_df(data_root: str,
                   int_dtype: type = np.int8):
    """df = get_cl_meta_df('./data/')

    This function loads the metadata for cell lines, process it and return
    as a dataframe. The processing includes:
        * change column names to ['data_src', 'site', 'type', 'category'];
        * remove the '-' in cell line names;
        * convert data types for more compact structure;

    Note that if the dataframe is already stored in the processed folder,
    the function simply read from file and return after converting dtypes.

    Args:
        data_root (str): path to the data root folder.
        int_dtype (type): int dtype for storage in RAM.

    Returns:
        pd.DataFrame: processed cell line metadata dataframe.
    """

    df_filename = 'cl_meta_df.pkl'
    df_path = os.path.join(data_root, PROC_FOLDER, df_filename)

    # If the dataframe already exists, load and continue ######################
    if os.path.exists(df_path):
        df = pd.read_pickle(df_path)

    # Otherwise load from raw files, process it and save ######################
    else:
        logger.debug('Processing cell line meta dataframe ... ')

        # Download the raw file if not exist
        download_files(filenames=CL_METADATA_FILENAME,
                       target_folder=os.path.join(data_root, RAW_FOLDER))

        df = pd.read_csv(
            os.path.join(data_root, RAW_FOLDER, CL_METADATA_FILENAME),
            sep='\t',
            header=0,
            index_col=0,
            usecols=['sample_name',
                     'dataset',
                     'simplified_tumor_site',
                     'simplified_tumor_type',
                     'sample_category'],
            dtype=str)

        # Renaming columns for shorter and better column names
        df.index.names = ['sample']
        df.columns = ['data_src', 'site', 'type', 'category']

        # Delete '-', which could be inconsistent between seq and meta
        print(df.shape)
        df.index = df.index.str.replace('-', '')
        print(df.shape)

        # Convert all the categorical data from text to numeric
        columns = df.columns
        dict_names = [i + '_dict.txt' for i in columns]
        for col, dict_name in zip(columns, dict_names):
            df[col] = encode_label_to_int(data_root=data_root,
                                          dict_name=dict_name,
                                          labels=df[col])

        # Convert data type into generic python types
        df = df.astype(int)

        # save to disk for future usage
        try:
            os.makedirs(os.path.join(data_root, PROC_FOLDER))
        except FileExistsError:
            pass
        df.to_pickle(df_path)

    df = df.astype(int_dtype)
    return df


if __name__ == '__main__':

    logging.basicConfig(level=logging.DEBUG)

    print('=' * 80 + '\nRNA sequence dataframe head:')
    print(get_rna_seq_df(data_root='../../data/',
                         rnaseq_feature_usage='source_scale',
                         rnaseq_scaling='std').head())

    print('=' * 80 + '\nCell line metadata dataframe head:')
    print(get_cl_meta_df(data_root='../../data/').head())
