from __future__ import absolute_import
from data_utils import get_file
# from six.moves import cPickle

import gzip
import logging
import os
import sys
import multiprocessing
import threading

import numpy as np
import pandas as pd

from itertools import cycle, islice

from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler


logger = logging.getLogger(__name__)

SEED = 2016

np.set_printoptions(threshold=np.nan)
np.random.seed(SEED)


def scale(df, scaling=None):
    """Scale data included in pandas dataframe.

    Parameters
    ----------
    df : pandas dataframe
        dataframe to scale
    scaling : 'maxabs', 'minmax', 'std', or None, optional (default 'std')
        type of scaling to apply
    """

    if scaling is None or scaling == 'None':
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
    df = pd.DataFrame(mat, columns=df.columns)

    return df


def impute_and_scale(df, scaling='std'):
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

    if scaling is None:
        return pd.DataFrame(mat, columns=df.columns)

    if scaling == 'maxabs':
        scaler = MaxAbsScaler()
    elif scaling == 'minmax':
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()

    mat = scaler.fit_transform(mat)

    df = pd.DataFrame(mat, columns=df.columns)

    return df


def load_cellline_expressions(path, ncols=None, scaling='std'):
    """Load cell line expression data, sub-select columns of gene expression
        randomly if specificed, scale the selected data and return a
        pandas dataframe.

    Parameters
    ----------
    path: string
        path to 'RNA_5_Platform_Gene_Transcript_Averaged_intensities.transposed.txt'
    ncols : int or None
        number of columns (gene expression) to randomly subselect (default None : use all data)
    scaling : 'maxabs' [-1,1], 'minmax' [0,1], 'std', or None, optional (default 'std')
        type of scaling to apply
    """

    df = pd.read_csv(path, sep='\t', engine='c',
                     na_values=['na','-',''])

    df1 = df['CellLine']
    df1 = df1.map(lambda x: x.replace('.', ':'))
    df1.name = 'CELLNAME'

    df2 = df.drop('CellLine', 1)

    total = df2.shape[1]
    if ncols and ncols < total:
        usecols = np.random.choice(total, size=ncols, replace=False)
        df2 = df2.iloc[:, usecols]

    df2 = impute_and_scale(df2, scaling)
    df2 = df2.astype(np.float32)
    df = pd.concat([df1, df2], axis=1)

    return df


def load_drug_descriptors(path, ncols=None, scaling='std'):
    """Load drug descriptor data, sub-select columns of drugs descriptors
        randomly if specificed, impute and scale the selected data, and return a
        pandas dataframe.

    Parameters
    ----------
    path: string
        path to 'descriptors.2D-NSC.5dose.filtered.txt'
    ncols : int or None
        number of columns (drugs descriptors) to randomly subselect (default None : use all data)
    scaling : 'maxabs' [-1,1], 'minmax' [0,1], 'std', or None, optional (default 'std')
        type of scaling to apply
    """

    df = pd.read_csv(path, sep='\t', engine='c',
                     na_values=['na','-',''],
                     dtype=np.float32)

    df1 = pd.DataFrame(df.loc[:,'NAME'].astype(int).astype(str))
    df1.rename(columns={'NAME': 'NSC'}, inplace=True)

    df2 = df.drop('NAME', 1)

    # # Filter columns if requested

    total = df2.shape[1]
    if ncols and ncols < total:
        usecols = np.random.choice(total, size=ncols, replace=False)
        df2 = df2.iloc[:,usecols]

    df2 = impute_and_scale(df2, scaling)
    df2 = df2.astype(np.float32)

    df_dg = pd.concat([df1, df2], axis=1)

    return df_dg


def load_drug_autoencoded(path, ncols=None, scaling='std'):
    """Load drug latent representation from autoencoder, sub-select
    columns of drugs randomly if specificed, impute and scale the
    selected data, and return a pandas dataframe.

    Parameters
    ----------
    path: string
        path to 'Aspuru-Guzik_NSC_latent_representation_292D.csv'
    ncols : int or None
        number of columns (drug latent representations) to randomly subselect (default None : use all data)
    scaling : 'maxabs' [-1,1], 'minmax' [0,1], 'std', or None, optional (default 'std')
        type of scaling to apply

    """

    df = pd.read_csv(path, engine='c', dtype=np.float32)

    df1 = pd.DataFrame(df.loc[:, 'NSC'].astype(int).astype(str))
    df2 = df.drop('NSC', 1)

    total = df2.shape[1]
    if ncols and ncols < total:
        usecols = np.random.choice(total, size=ncols, replace=False)
        df2 = df2.iloc[:, usecols]

    df2 = impute_and_scale(df2, scaling)
    df2 = df2.astype(np.float32)

    df = pd.concat([df1, df2], axis=1)

    return df


def load_dose_response(path, min_logconc=-5., max_logconc=-5., subsample=None):
    """Load cell line response to different drug compounds, sub-select response for a specific
        drug log concentration range and return a pandas dataframe.

    Parameters
    ----------
    path: string
        path to 'NCI60_dose_response_with_missing_z5_avg.csv'
    min_logconc : -3, -4, -5, -6, -7, optional (default -5)
        min log concentration of drug to return cell line growth
    max_logconc : -3, -4, -5, -6, -7, optional (default -5)
        max log concentration of drug to return cell line growth
    subsample: None, 'naive_balancing' (default None)
        subsampling strategy to use to balance the data based on growth
    """

    df = pd.read_csv(path, sep=',', engine='c',
                     na_values=['na','-',''],
                     dtype={'NSC':object, 'CELLNAME':str, 'LOG_CONCENTRATION':np.float32, 'GROWTH':np.float32})

    df = df[(df['LOG_CONCENTRATION'] >= min_logconc) & (df['LOG_CONCENTRATION'] <= max_logconc)]

    df = df[['NSC', 'CELLNAME', 'GROWTH', 'LOG_CONCENTRATION']]

    if subsample and subsample == 'naive_balancing':
        df1 = df[df['GROWTH'] <= 0]
        df2 = df[(df['GROWTH'] > 0) & (df['GROWTH'] < 50)].sample(frac=0.7, random_state=SEED)
        df3 = df[(df['GROWTH'] >= 50) & (df['GROWTH'] <= 100)].sample(frac=0.18, random_state=SEED)
        df4 = df[df['GROWTH'] > 100].sample(frac=0.01, random_state=SEED)
        df = pd.concat([df1, df2, df3, df4])

    df = df.set_index(['NSC'])

    return df


class RegressionDataGenerator(object):
    """Generate merged drug response, drug descriptors and cell line essay data
    """

    def __init__(self, val_split=0.2, shuffle=True, drug_features='descriptors',
                 feature_subsample=None, scaling='std', scramble=False,
                 min_logconc=-5., max_logconc=-4., subsample='naive_balancing',
                 category_cutoffs=[0.]):
        """Initialize data merging drug response, drug descriptors and cell line essay.
           Shuffle and split training and validation set

        Parameters
        ----------
        val_split : float, optional (default 0.2)
            percentage of data to use in validation
        shuffle : True or False, optional (default True)
            if True shuffles the merged data before splitting training and validation sets
        drug_features: 'descriptors', 'latent', 'both', 'noise' (default 'descriptors')
            use dragon7 descriptors, latent representations from Aspuru-Guzik's SMILES autoencoder trained on NSC drugs, or both; use random features if set to noise
        feature_subsample: None or integer (default None)
            number of feature columns to use from cellline expressions and drug descriptors
        scaling: None, 'std', 'minmax' or 'maxabs' (default 'std')
            type of feature scaling: 'maxabs' to [-1,1], 'maxabs' to [-1, 1], 'std' for standard normalization
        scramble: True or False, optional (default False)
            if True randomly shuffle dose response data as a control
        min_logconc: float value between -3 and -7, optional (default -5.)
            min log concentration of drug to return cell line growth
        max_logconc: float value between -3 and -7, optional (default -4.)
            max log concentration of drug to return cell line growth
        subsample: 'naive_balancing' or None
            if True balance dose response data with crude subsampling
        category_cutoffs: list of floats (between -1 and +1) (default None)
            growth thresholds seperating non-response and response categories
        """

        self.lock = threading.Lock()
        self.drug_features = drug_features

        server = 'http://ftp.mcs.anl.gov/pub/candle/public/benchmarks/P1B3/'

        cell_expr_path = get_file('P1B3_cellline_expressions.tsv', origin=server+'P1B3_cellline_expressions.tsv')
        drug_desc_path = get_file('P1B3_drug_descriptors.tsv', origin=server+'P1B3_drug_descriptors.tsv')
        drug_auen_path = get_file('P1B3_drug_latent.csv', origin=server+'P1B3_drug_latent.csv')
        dose_resp_path = get_file('P1B3_dose_response.csv', origin=server+'P1B3_dose_response.csv')
        test_cell_path = get_file('P1B3_test_celllines.txt', origin=server+'P1B3_test_celllines.txt')
        test_drug_path = get_file('P1B3_test_drugs.txt', origin=server+'P1B3_test_drugs.txt')

        self.df_cellline = load_cellline_expressions(cell_expr_path, ncols=feature_subsample, scaling=scaling)

        df = load_dose_response(dose_resp_path, min_logconc=min_logconc, max_logconc=max_logconc, subsample=subsample)
        logger.info('Loaded {} unique (D, CL) response sets.'.format(df.shape[0]))
        # df[['GROWTH', 'LOG_CONCENTRATION']].to_csv('all.response.csv')

        df = df.reset_index()
        df = df.merge(self.df_cellline[['CELLNAME']], on='CELLNAME')

        if drug_features in ['descriptors', 'both']:
            self.df_drug_desc = load_drug_descriptors(drug_desc_path, ncols=feature_subsample, scaling=scaling)
            df = df.merge(self.df_drug_desc[['NSC']], on='NSC')
        if drug_features in ['latent', 'both']:
            self.df_drug_auen = load_drug_autoencoded(drug_auen_path, ncols=feature_subsample, scaling=scaling)
            df = df.merge(self.df_drug_auen[['NSC']], on='NSC')
        if drug_features == 'noise':
            df_drug_ids = df[['NSC']].drop_duplicates()
            noise = np.random.normal(size=(df_drug_ids.shape[0], 500))
            df_rand = pd.DataFrame(noise, index=df_drug_ids['NSC'],
                                   columns=['RAND-{:03d}'.format(x) for x in range(500)])
            self.df_drug_rand = df_rand.reset_index()

        logger.debug('Filltered down to {} rows with matching information.'.format(df.shape[0]))
        # df[['GROWTH', 'LOG_CONCENTRATION']].to_csv('filtered.response.csv')

        df_test_cell = pd.read_csv(test_cell_path)
        df_test_drug = pd.read_csv(test_drug_path, dtype={'NSC':object})

        df_test = df.merge(df_test_cell, on='CELLNAME').merge(df_test_drug, on='NSC')
        logger.debug('Set aside {} rows as test data'.format(df_test.shape[0]))

        df_train_val = df[(~df['NSC'].isin(df_test_drug['NSC'])) & (~df['CELLNAME'].isin(df_test_cell['CELLNAME']))]
        logger.debug('Combined train and test set has {} rows'.format(df_train_val.shape[0]))

        if shuffle:
            df_train_val = df_train_val.sample(frac=1.0, random_state=SEED)
            df_test = df_test.sample(frac=1.0, random_state=SEED)

        self.df_response = pd.concat([df_train_val, df_test]).reset_index(drop=True)

        if scramble:
            growth = self.df_response[['GROWTH']]
            random_growth = growth.iloc[np.random.permutation(np.arange(growth.shape[0]))].reset_index()
            self.df_response[['GROWTH']] = random_growth['GROWTH']
            logger.warn('Randomly shuffled dose response growth values.')

        logger.info('Distribution of dose response:')
        logger.info(self.df_response[['GROWTH']].describe())

        if category_cutoffs is not None:
            growth = self.df_response['GROWTH']
            classes = np.digitize(growth, category_cutoffs)
            bc = np.bincount(classes)
            min_g = np.min(growth) / 100
            max_g = np.max(growth) / 100
            logger.info('Category cutoffs: {}'.format(category_cutoffs))
            logger.info('Dose response bin counts:')
            for i, count in enumerate(bc):
                lower = min_g if i == 0 else category_cutoffs[i-1]
                upper = max_g if i == len(bc)-1 else category_cutoffs[i]
                logger.info('  Class {}: {:7d} ({:.4f}) - between {:+.2f} and {:+.2f}'.
                            format(i, count, count/len(growth), lower, upper))
            logger.info('  Total: {:9d}'.format(len(growth)))

        nrows = df_train_val.shape[0]
        self.n_test = df_test.shape[0]
        self.n_val = int(nrows * val_split)
        self.n_train = nrows - self.n_val

        self.cycle_train = cycle(range(nrows - self.n_val))
        self.cycle_val = cycle(range(nrows)[-self.n_val:])
        self.cycle_test = cycle(range(nrows, nrows + self.n_test))
        logger.info('Rows in train: {}, val: {}, test: {}'.format(self.n_train, self.n_val, self.n_test))

        self.input_dim = self.df_cellline.shape[1] - 1 + 1  # remove CELLNAME; add concentration
        logger.info('Features:')
        logger.info('  concentration: 1')
        logger.info('  cell line expression: {}'.format(self.input_dim-1))
        if self.drug_features in ['descriptors', 'both']:
            self.input_dim += self.df_drug_desc.shape[1] - 1  # remove NSC
            logger.info('  drug descriptors: {}'.format(self.df_drug_desc.shape[1] - 1))
        if self.drug_features in ['latent', 'both']:
            self.input_dim += self.df_drug_auen.shape[1] - 1  # remove NSC
            logger.info('  drug latent representations: {}'.format(self.df_drug_auen.shape[1] - 1))
        if self.drug_features == 'noise':
            self.input_dim += self.df_drug_rand.shape[1] - 1  # remove NSC
            logger.info('  drug random vectors: {}'.format(self.df_drug_rand.shape[1] - 1))
        logger.info('Total input dimensions: {}'.format(self.input_dim))

    def flow(self, batch_size=32, data='train', topology=None):
        if data == 'val':
            cyc = self.cycle_val
        elif data == 'test':
            cyc = self.cycle_test
        else:
            cyc = self.cycle_train

        while 1:
            self.lock.acquire()
            indices = list(islice(cyc, batch_size))
            # print("\nProcess: {}, Batch indices start: {}".format(multiprocessing.current_process().name, indices[0]))
            self.lock.release()

            df = self.df_response.iloc[indices, :]
            df = pd.merge(df, self.df_cellline, on='CELLNAME')

            if self.drug_features in ['descriptors', 'both']:
                df = df.merge(self.df_drug_desc, on='NSC')
            if self.drug_features in ['latent', 'both']:
                df = df.merge(self.df_drug_auen, on='NSC')
            if self.drug_features == 'noise':
                df = df.merge(self.df_drug_rand, on='NSC')

            df = df.drop(['CELLNAME', 'NSC'], 1)
            x = np.array(df.iloc[:, 1:])
            y = np.array(df.iloc[:, 0])
            y = y / 100.

            if topology == 'simple_local':
                yield x.reshape(x.shape + (1,)), y
                # yield x.reshape(x.shape[0], 1, x.shape[1]), y
            else:
                yield x, y
