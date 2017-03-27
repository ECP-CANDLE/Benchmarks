#! /usr/bin/env python

"""Multilayer Perceptron for drug response problem"""

from __future__ import division, print_function

from itertools import cycle, islice

import csv
import logging

import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import Callback, ModelCheckpoint

from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler


# For non-interactive plotting
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

# For logging model parameters
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)


# Model and Training parameters

# Seed for random generation
SEED = 2016
# Size of batch for training
BATCH_SIZE = 100
# Number of training epochs
NB_EPOCH = 20

# Percentage of drop used in training
DROP = 0.1
# Activation function (options: 'relu', 'tanh', 'sigmoid', 'hard_sigmoid',
#                               'linear')
ACTIVATION = 'relu'

# Type of feature scaling (options: 'maxabs': to [-1,1]
#                                   'minmax': to [0,1]
#                                   None    : standard normalization
SCALING = 'minmax'
# Features to (randomly) sample from cell lines or drug descriptors
FEATURE_SUBSAMPLE = 500#0

# Number of units per layer
L1 = 1000
L2 = 500
L3 = 100
L4 = 50
LAYERS = [L1, L2, L3, L4]


np.set_printoptions(threshold=np.nan)
np.random.seed(SEED)


def extension_from_parameters():
    """Construct string for saving model with annotation of parameters"""
    ext = ''
    ext += '.A={}'.format(ACTIVATION)
    ext += '.B={}'.format(BATCH_SIZE)
    ext += '.D={}'.format(DROP)
    ext += '.E={}'.format(NB_EPOCH)
    if FEATURE_SUBSAMPLE:
        ext += '.F={}'.format(FEATURE_SUBSAMPLE)
    for i, n in enumerate(LAYERS):
        if n:
            ext += '.L{}={}'.format(i+1, n)

    ext += '.S={}'.format(SCALING)

    return ext


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


def load_cellline_expressions(ncols=None, scaling='minmax'):
    """Load cell line expression data, sub-select columns of gene expression
        randomly if specificed, scale the selected data and return a
        pandas dataframe.
        
    Parameters
    ----------
    ncols : int or None
        number of columns (gene expression) to randomly subselect (default None : use all data)
    scaling : 'maxabs' [-1,1], 'minmax' [0,1], 'std', or None, optional (default 'minmax')
        type of scaling to apply
    """

    df = pd.read_csv('RNA__5_Platform_Gene_Transcript_Averaged_intensities.csv',
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


def load_drug_descriptors(ncols=None, scaling='minmax'):
    """Load drug descriptor data, sub-select columns of drugs descriptors
        randomly if specificed, impute and scale the selected data, and return a
        pandas dataframe.
        
    Parameters
    ----------
    ncols : int or None
        number of columns (drugs descriptors) to randomly subselect (default None : use all data)
    scaling : 'maxabs' [-1,1], 'minmax' [0,1], 'std', or None, optional (default 'minmax')
        type of scaling to apply
    """

    df = pd.read_csv('descriptors.2D-NSC.5dose.filtered.txt', sep='\t',
                     na_values=['na','-',''],
                     dtype=np.float32,
                     engine='c')
        
    df1 = pd.DataFrame(df.loc[:,'NAME'].astype(int).astype(str))
    df1.rename(columns={'NAME': 'NSC'}, inplace=True)

    df2 = df.drop('NAME', 1)
    
    # Filter columns if requested
    if ncols:
        #usecols = list(range(ncols))
        total = df2.shape[1]
        usecols = np.random.choice(total, size=ncols, replace=False)
        df2 = df2.iloc[:,usecols]


    df2 = impute_and_scale(df2, scaling)
    df2 = df2.astype(np.float32)


    df_dg = pd.concat([df1, df2], axis=1)

    return df_dg



def load_drug_response(concentration=-5.):
    """Load cell line response to different drug compounds, sub-select response for a specific
        drug log concentration and return a pandas dataframe.
        
    Parameters
    ----------
    concentration : -3, -4, -5, -6, -7, optional (default -5)
        log concentration of drug to return cell line growth
    """

    df_response = pd.read_csv('NCI60_dose_response_with_missing_z5_avg.csv', sep=',',
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

    def __init__(self, val_split=0.2, shuffle=True):
        """Initialize data merging drug response, drug descriptors and cell line essay.
           Shuffle and split training and validation set
            
        Parameters
        ----------
        val_split : float, optional (default 0.2)
            percentage of data to use in validation
        shuffle : True or False, optional (default True)
            if True shuffles the merged data before splitting training and validation sets
        """

        self.df_cellline = load_cellline_expressions(FEATURE_SUBSAMPLE, scaling=SCALING)
        self.df_drug = load_drug_descriptors(FEATURE_SUBSAMPLE, scaling=SCALING)

        df_drug_response = load_drug_response()
        self.df_response = df_drug_response.reset_index()
        
        if shuffle:
            self.df_response = self.df_response.sample(frac=1.0)
            
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


def plot_error(model, generator, samples, batch, file_ext):
    if batch % 20:
        return
    fig2 = plt.figure()
    samples = 100
    diffs = None
    y_all = None
    for i in range(samples):
        X, y = next(generator)
        y = y.ravel() * 100
        y_pred = model.predict(X)
        y_pred = y_pred.ravel() * 100
        diff =  y - y_pred
        diffs = np.concatenate((diffs, diff)) if diffs is not None else diff
        y_all = np.concatenate((y_all, y)) if y_all is not None else y
    bins = np.linspace(-200, 200, 100)
    if batch == 0:
        y_shuf = np.random.permutation(y_all)
        plt.hist(y_all - y_shuf, bins, alpha=0.5, label='Random')
    #plt.hist(diffs, bins, alpha=0.35-batch/100., label='Epoch {}'.format(batch+1))
    plt.hist(diffs, bins, alpha=0.3, label='Epoch {}'.format(batch+1))
    plt.title("Histogram of errors in percentage growth")
    plt.legend(loc='upper right')
    plt.savefig('plot'+file_ext+'.b'+str(batch)+'.png')
    # plt.savefig('plot'+file_ext+'.png')
    plt.close()

    # Plot measured vs. predicted values
    fig, ax = plt.subplots()
    plt.grid('on')
    ax.scatter(y, y_pred, color='red')
    ax.plot([y.min(), y.max()],
            [y.min(), y.max()], 'k--', lw=4)
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    #plt.show()
    plt.savefig('meas_vs_pred'+file_ext+'.b'+str(batch)+'.png')
    plt.close()



class BestLossHistory(Callback):
    def __init__(self, generator, samples, ext):
        super(BestLossHistory, self).__init__()
        self.generator = generator
        self.samples = samples
        self.ext = ext

    def on_train_begin(self, logs={}):
        self.best_val_loss = np.Inf
        self.best_val_acc = -np.Inf
        self.best_model = None

    def on_epoch_end(self, batch, logs={}):
        if float(logs.get('val_loss', 0)) < self.best_val_loss:
            self.best_model = self.model
            plot_error(self.best_model, self.generator, self.samples, batch, self.ext)
        self.best_val_loss = min(float(logs.get('val_loss', 0)), self.best_val_loss)
        self.best_val_acc = max(float(logs.get('val_acc', 0)), self.best_val_acc)


def main():
    ext = extension_from_parameters()

    out_dim = 1
    loss = 'mse'
    metrics = None
    #metrics = ['accuracy'] if CATEGORICAL else None

    datagen = RegressionDataGenerator()
    train_gen = datagen.flow(batch_size=BATCH_SIZE)
    val_gen = datagen.flow(val=True, batch_size=BATCH_SIZE)
    val_gen2 = datagen.flow(val=True, batch_size=BATCH_SIZE)

    model = Sequential()
    for layer in LAYERS:
        if layer:
            model.add(Dense(layer, input_dim=datagen.input_dim, activation=ACTIVATION))
            if DROP:
                model.add(Dropout(DROP))
    model.add(Dense(out_dim))

    model.summary()
    model.compile(loss=loss, optimizer='rmsprop', metrics=metrics)

    train_samples = int(datagen.n_train/BATCH_SIZE) * BATCH_SIZE
    val_samples = int(datagen.n_val/BATCH_SIZE) * BATCH_SIZE

    history = BestLossHistory(val_gen2, val_samples, ext)
    checkpointer = ModelCheckpoint(filepath='model'+ext+'.h5', save_best_only=True)

    model.fit_generator(train_gen, train_samples,
                        nb_epoch = NB_EPOCH,
                        validation_data = val_gen,
                        nb_val_samples = val_samples,
                        callbacks=[history, checkpointer])
                        # nb_worker = 1)


if __name__ == '__main__':
    main()
