from __future__ import absolute_import
from data_utils import get_file
# from six.moves import cPickle

import gzip
import logging
import os
import sys

import numpy as np
import pandas as pd

from itertools import cycle, islice

from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler


# For logging model parameters
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)


seed = 2016


def scale(df, scaling=None):
    """Scale data included in pandas dataframe.

    Parameters
    ----------
    df : pandas dataframe
        dataframe to scale
    scaling : 'maxabs', 'minmax', 'std', or None, optional (default 'std')
        type of scaling to apply
    """

    if scaling is None:
        return df

    df = df.dropna(axis=1, how='any')

    # Scaling data
    if scaling == 'maxabs':
        # Normalizing -1 to 1
        scaler = MaxAbsScaler()
    elif scaling == 'minmax':
        # Scaling to [0,1]
        scaler = MinMaxScaler()
    else:
        # Standard normalization
        scaler = StandardScaler()

    mat = df.as_matrix()
    mat = scaler.fit_transform(mat)
    # print(mat.shape)
    df = pd.DataFrame(mat, columns=df.columns)

    return df


def impute_and_scale(df, scaling=None):
    """Impute missing values with mean and scale data included in pandas dataframe.

    Parameters
    ----------
    df : pandas dataframe
        dataframe to impute and scale
    scaling : 'maxabs' [-1,1], 'minmax' [0,1], 'std', or None, optional (default 'std')
        type of scaling to apply
    """

    df = df.dropna(axis=1, how='all')

    imputer = Imputer(strategy='mean', axis=0)
    mat = imputer.fit_transform(df)
    # print(mat.shape)

    if scaling is None:
        return pd.DataFrame(mat, columns=df.columns)

    # Scaling data
    if scaling == 'maxabs':
        # Normalizing -1 to 1
        scaler = MaxAbsScaler()
    elif scaling == 'minmax':
        # Scaling to [0,1]
        scaler = MinMaxScaler()
    else:
        # Standard normalization
        scaler = StandardScaler()

    mat = scaler.fit_transform(mat)

    # print(mat.shape)
    df = pd.DataFrame(mat, columns=df.columns)

    return df


def load_cellline_expressions(path, ncols=None, scaling='minmax'):
    """Load cell line expression data, sub-select columns of gene expression
        randomly if specificed, scale the selected data and return a
        pandas dataframe.

    Parameters
    ----------
    path: string
        path to 'RNA_5_Platform_Gene_Transcript_Averaged_intensities.transposed.txt'
    ncols : int or None
        number of columns (gene expression) to randomly subselect (default None : use all data)
    scaling : 'maxabs' [-1,1], 'minmax' [0,1], 'std', or None, optional (default 'minmax')
        type of scaling to apply
    """

    usecols = list(range(ncols)) if ncols else None
    df = pd.read_csv(path,
                     na_values=['na','-',''],
                     usecols=usecols,
                     sep='\t', engine='c')

    # df = df.drop('CNS.SF_539')
    # df = df.dropna(how='any')        # No imputation of data

    df1 = df['CellLine']
    df1 = df1.map(lambda x: x.replace('.', ':'))
    df1.name = 'CELLNAME'

    df2 = df.drop('CellLine', 1)
    df2 = impute_and_scale(df2, scaling)
    df2 = df2.astype(np.float32)
    df = pd.concat([df1, df2], axis=1)

    return df


def load_cellline_expressions_CGC(path, ncols=None, scaling='minmax'):
    """Load cell line expression data, sub-select columns of gene expression
        randomly if specificed, scale the selected data and return a
        pandas dataframe.

    Parameters
    ----------
    path: string
        path to 'RNA_5_Platform_Gene_Transcript_Averaged_intensities.csv'
    ncols : int or None
        number of columns (gene expression) to randomly subselect (default None : use all data)
    scaling : 'maxabs' [-1,1], 'minmax' [0,1], 'std', or None, optional (default 'minmax')
        type of scaling to apply
    """

    df = pd.read_csv(path,
                     na_values=['na','-',''],
                     dtype={'Gene name': str },
                     sep=',', engine='c')

    df = df.drop('CNS:SF_539', 1)    # Drop very incomplete cell line
    df = df.dropna(how='any')        # No imputation of data
    #geneName = list(df['Gene name'])
    geneName = df['Gene name']
    df = df.drop('Gene name', 1)     # Drop names of corresponding genes
    df = df.T                        # Transpose data to have cell lines per row

    df1 = pd.DataFrame(df.index)
    df2 = df

    if ncols:
        total = df2.shape[1]
        usecols = np.random.choice(total, size=ncols, replace=False)
        df2 = df2.iloc[:,usecols]
        geneName = geneName[usecols]

    df2 = scale(df2, scaling)
    df2 = df2.astype(np.float32)

    df = pd.concat([df1, df2], axis=1)
    geneName = list(geneName)
    geneName.insert(0, 'CELLNAME')
    df.columns = geneName

    return df


def load_drug_descriptors(path, ncols=None, scaling='minmax'):
    """Load drug descriptor data, sub-select columns of drugs descriptors
        randomly if specificed, impute and scale the selected data, and return a
        pandas dataframe.

    Parameters
    ----------
    path: string
        path to 'descriptors.2D-NSC.5dose.filtered.txt'
    ncols : int or None
        number of columns (drugs descriptors) to randomly subselect (default None : use all data)
    scaling : 'maxabs' [-1,1], 'minmax' [0,1], 'std', or None, optional (default 'minmax')
        type of scaling to apply
    """

    usecols = list(range(ncols)) if ncols else None
    df = pd.read_csv(path, sep='\t',
                     na_values=['na','-',''],
                     dtype=np.float32,
                     usecols=usecols,
                     engine='c')

    df1 = pd.DataFrame(df.loc[:,'NAME'].astype(int).astype(str))
    df1.rename(columns={'NAME': 'NSC'}, inplace=True)

    df2 = df.drop('NAME', 1)

    # # Filter columns if requested
    # if ncols:
    #     #usecols = list(range(ncols))
    #     total = df2.shape[1]
    #     usecols = np.random.choice(total, size=ncols, replace=False)
    #     df2 = df2.iloc[:,usecols]

    df2 = impute_and_scale(df2, scaling)
    df2 = df2.astype(np.float32)

    df_dg = pd.concat([df1, df2], axis=1)

    return df_dg


def load_drug_response(path, concentration=-5.):
    """Load cell line response to different drug compounds, sub-select response for a specific
        drug log concentration and return a pandas dataframe.

    Parameters
    ----------
    path: string
        path to 'NCI60_dose_response_with_missing_z5_avg.csv'
    concentration : -3, -4, -5, -6, -7, optional (default -5)
        log concentration of drug to return cell line growth
    """

    df_response = pd.read_csv(path, sep=',',
                              na_values=['na','-',''],
                              dtype={'NSC':object, 'CELLNAME':str, 'LOG_CONCENTRATION':np.float32,
                                     'GROWTH':np.float32 },
                              engine='c')

    df_logconc = df_response.loc[df_response['LOG_CONCENTRATION'] == concentration]

    # Sub select columns
    df_to_use = df_logconc[['NSC', 'CELLNAME', 'GROWTH']]
    df_to_use = df_to_use.set_index(['NSC'])

    return df_to_use


class RegressionDataGenerator(object):
    """Generate merged drug response, drug descriptors and cell line essay data
    """

    def __init__(self, val_split=0.2, shuffle=True, feature_subsample=None, scaling=None, concentration=-5.):
        """Initialize data merging drug response, drug descriptors and cell line essay.
           Shuffle and split training and validation set

        Parameters
        ----------
        val_split : float, optional (default 0.2)
            percentage of data to use in validation
        shuffle : True or False, optional (default True)
            if True shuffles the merged data before splitting training and validation sets
        feature_subsample: None or integer (default None)
            number of feature columns to use from cellline expressions and drug descriptors
        scaling: None, 'minmax' or 'maxabs' (default 'minmax')
            type of feature scaling: 'maxabs' to [-1,1], 'maxabs' to [-1, 1], None for standard normalization
        concentration : -3, -4, -5, -6, -7, optional (default -5)
            log concentration of drug to return cell line growth
        """

        cell_expr_path = get_file('RNA_5_Platform_Gene_Transcript_Averaged_intensities.transposed.txt',
                                  origin='http://bioseed.mcs.anl.gov/~fangfang/p1b3/RNA_5_Platform_Gene_Transcript_Averaged_intensities.transposed.txt')
        drug_desc_path = get_file('descriptors.2D-NSC.5dose.filtered.txt',
                                  origin='http://bioseed.mcs.anl.gov/~fangfang/p1b3/descriptors.2D-NSC.5dose.filtered.txt')
        drug_resp_path = get_file('NCI60_dose_response_with_missing_z5_avg.csv',
                                  origin='http://bioseed.mcs.anl.gov/~fangfang/p1b3/NCI60_dose_response_with_missing_z5_avg.csv')

        self.df_cellline = load_cellline_expressions(cell_expr_path, ncols=feature_subsample, scaling=scaling)
        self.df_drug = load_drug_descriptors(drug_desc_path, ncols=feature_subsample, scaling=scaling)

        df_drug_response = load_drug_response(drug_resp_path, concentration=concentration)
        self.df_response = df_drug_response.reset_index()

        if shuffle:
            self.df_response = self.df_response.sample(frac=1.0, random_state=seed)

        nrows = self.df_response.shape[0]
        logger.info('Loaded {} unique (D, CL) response sets.'.format(nrows))
        logger.info(self.df_response['GROWTH'].describe())
        self.n_val = int(nrows * val_split)
        self.n_train = nrows - self.n_val
        self.cycle_train = cycle(range(nrows - self.n_val))
        self.cycle_val = cycle(range(nrows)[-self.n_val:])
        self.input_dim = self.df_cellline.shape[1] + self.df_drug.shape[1] - 2
        # print(nrows, self.n_train, self.n_val)
        logger.info('Input dim = {}'.format(self.input_dim))

    def flow(self, batch_size=32, val=False):
        cyc = self.cycle_val if val else self.cycle_train
        while 1:
            indices = list(islice(cyc, batch_size))
            df = self.df_response.iloc[indices, :]
            df = pd.merge(df, self.df_cellline, on='CELLNAME')
            df = pd.merge(df, self.df_drug, on='NSC')
            df = df.drop(['CELLNAME', 'NSC'], 1)
            X = np.array(df.iloc[:, 1:])
            y = np.array(df.iloc[:, 0])
            y = y / 100.
            yield X, y
