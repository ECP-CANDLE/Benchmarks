from __future__ import absolute_import

import numpy as np
import pandas as pd

from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler

from keras.utils import np_utils

from default_utils import DEFAULT_SEED
from default_utils import DEFAULT_DATATYPE

def convert_to_class(y_one_hot, dtype=int):

    maxi = lambda a: a.argmax()
    iter_to_na = lambda i: np.fromiter(i, dtype=dtype)
    return np.array([maxi(a) for a in y_one_hot])


def scale_array(mat, scaling=None):
    """Scale data included in numpy array.
        
        Parameters
        ----------
        mat : numpy array
            array to scale
        scaling : 'maxabs', 'minmax', 'std', or None, optional (default 'None')
            type of scaling to apply
    """
    
    if scaling is None or scaling.lower() == 'none':
        return mat

    # Scaling data
    if scaling == 'maxabs':
        # Normalizing -1 to 1
        scaler = MaxAbsScaler(copy=False)
    elif scaling == 'minmax':
        # Scaling to [0,1]
        scaler = MinMaxScaler(copy=False)
    else:
        # Standard normalization
        scaler = StandardScaler(copy=False)
    
    return scaler.fit_transform(mat)



def impute_and_scale_array(mat, scaling=None):
    """Impute missing values with mean and scale data included in numpy array.
        
        Parameters
        ----------
        mat : numpy array
            array to scale
        scaling : 'maxabs', 'minmax', 'std', or None, optional (default 'None')
            type of scaling to apply
    """
    
    imputer = Imputer(strategy='mean', axis=0, copy=False)
    imputer.fit_transform(mat)
    #mat = imputer.fit_transform(mat)
    
    return scale_array(mat, scaling)



def load_X_data(train_file, test_file,
                drop_cols=None, n_cols=None, shuffle=False, scaling=None,
                dtype=DEFAULT_DATATYPE, seed=DEFAULT_SEED):

    # compensates for the columns to drop if there is a feature subselection
    usecols = list(range(n_cols + len(drop_cols))) if n_cols else None
        
    df_train = pd.read_csv(train_file, engine='c', usecols=usecols)
    df_test = pd.read_csv(test_file, engine='c', usecols=usecols)

    # Drop specified columns
    if drop_cols is not None:
        for col in drop_cols:
            df_train.drop(col, axis=1, inplace=True)
            df_test.drop(col, axis=1, inplace=True)

    if shuffle:
        df_train = df_train.sample(frac=1, random_state=seed)
        df_test = df_test.sample(frac=1, random_state=seed)

    X_train = df_train.values.astype(dtype)
    X_test = df_test.values.astype(dtype)

    mat = np.concatenate((X_train, X_test), axis=0)
    # Scale data
    if scaling is not None:
        mat = scale_array(mat, scaling)

    X_train = mat[:X_train.shape[0], :]
    X_test = mat[X_train.shape[0]:, :]

    return X_train, X_test


def load_X_data2(train_file, test_file,
                drop_cols=None, n_cols=None, shuffle=False, scaling=None,
                validation_split=0.1, dtype=DEFAULT_DATATYPE, seed=DEFAULT_SEED):

    # compensates for the columns to drop if there is a feature subselection
    usecols = list(range(n_cols + len(drop_cols))) if n_cols else None

    df_train = pd.read_csv(train_file, engine='c', usecols=usecols)
    df_test = pd.read_csv(test_file, engine='c', usecols=usecols)

    # Drop specified columns
    if drop_cols is not None:
        for col in drop_cols:
            df_train.drop(col, axis=1, inplace=True)
            df_test.drop(col, axis=1, inplace=True)

    if shuffle:
        df_train = df_train.sample(frac=1, random_state=seed)
        df_test = df_test.sample(frac=1, random_state=seed)

    X_train = df_train.values.astype(dtype)
    X_test = df_test.values.astype(dtype)

    mat = np.concatenate((X_train, X_test), axis=0)
    # Scale data
    if scaling is not None:
        mat = scale_array(mat, scaling)

    # Separate training in training and validation splits after scaling
    sizeTrain = X_train.shape[0]
    X_test = mat[sizeTrain:, :]
    numVal = int(sizeTrain * validation_split)
    X_val = mat[:numVal, :]
    X_train = mat[numVal:sizeTrain, :]

    return X_train, X_val, X_test


def load_Xy_one_hot_data(train_file, test_file,
                        class_col=None, drop_cols=None, n_cols=None, shuffle=False, scaling=None,
                        dtype=DEFAULT_DATATYPE, seed=DEFAULT_SEED):

    assert class_col != None
    
    # compensates for the columns to drop if there is a feature subselection
    usecols = list(range(n_cols + len(drop_cols))) if n_cols else None

    df_train = pd.read_csv(train_file, engine='c', usecols=usecols)
    df_test = pd.read_csv(test_file, engine='c', usecols=usecols)
    
    if shuffle:
        df_train = df_train.sample(frac=1, random_state=seed)
        df_test = df_test.sample(frac=1, random_state=seed)

    # Get class
    y_train = pd.get_dummies(df_train[class_col]).values
    y_test = pd.get_dummies(df_test[class_col]).values

    # Drop specified columns
    if drop_cols is not None:
        for col in drop_cols:
            df_train.drop(col, axis=1, inplace=True)
            df_test.drop(col, axis=1, inplace=True)


    # Convert from pandas dataframe to numpy array
    X_train = df_train.values.astype(dtype)
    print("X_train dtype: ", X_train.dtype)
    X_test = df_test.values.astype(dtype)
    print("X_test dtype: ", X_test.dtype)
    # Concatenate training and testing to scale data
    mat = np.concatenate((X_train, X_test), axis=0)
    print("mat dtype: ", mat.dtype)
    # Scale data
    if scaling is not None:
        mat = scale_array(mat, scaling)
    # Recover training and testing splits after scaling
    X_train = mat[:X_train.shape[0], :]
    X_test = mat[X_train.shape[0]:, :]

    return (X_train, y_train), (X_test, y_test)


def load_Xy_one_hot_data2(train_file, test_file,
                    class_col=None, drop_cols=None, n_cols=None, shuffle=False, scaling=None,
                    validation_split=0.1, dtype=DEFAULT_DATATYPE, seed=DEFAULT_SEED):
    
    assert class_col != None
    
    # compensates for the columns to drop if there is a feature subselection
    usecols = list(range(n_cols + len(drop_cols))) if n_cols else None
    
    df_train = pd.read_csv(train_file, engine='c', usecols=usecols)
    df_test = pd.read_csv(test_file, engine='c', usecols=usecols)

    if shuffle:
        df_train = df_train.sample(frac=1, random_state=seed)
        df_test = df_test.sample(frac=1, random_state=seed)

    # Get class
    y_train = pd.get_dummies(df_train[class_col]).values
    y_test = pd.get_dummies(df_test[class_col]).values
    
    # Drop specified columns
    if drop_cols is not None:
        for col in drop_cols:
            df_train.drop(col, axis=1, inplace=True)
            df_test.drop(col, axis=1, inplace=True)

    # Convert from pandas dataframe to numpy array
    X_train = df_train.values.astype(dtype)
    X_test = df_test.values.astype(dtype)
    # Concatenate training and testing to scale data
    mat = np.concatenate((X_train, X_test), axis=0)
    # Scale data
    if scaling is not None:
        mat = scale_array(mat, scaling)
    # Separate training in training and validation splits after scaling
    sizeTrain = X_train.shape[0]
    X_test = mat[sizeTrain:, :]
    numVal = int(sizeTrain * validation_split)
    X_val = mat[:numVal, :]
    X_train = mat[numVal:sizeTrain, :]
    # Analogously separate y in training in training and validation splits
    y_val = y_train[:numVal, :]
    y_train = y_train[numVal:sizeTrain, :]

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)



def load_Xy_data2(train_file, test_file, class_col=None, drop_cols=None, n_cols=None, shuffle=False, scaling=None,
                  validation_split=0.1, dtype=DEFAULT_DATATYPE, seed=DEFAULT_SEED):
    
    assert class_col != None
    
    (X_train, y_train_oh), (X_val, y_val_oh), (X_test, y_test_oh) = load_Xy_one_hot_data2(train_file, test_file,
                                                                                 class_col, drop_cols, n_cols, shuffle, scaling,
                                                                                 validation_split, dtype, seed)

    y_train = convert_to_class(y_train_oh)
    y_val = convert_to_class(y_val_oh)
    y_test = convert_to_class(y_test_oh)
    

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)



def load_Xy_data_noheader(train_file, test_file, classes, scaling=None, dtype=DEFAULT_DATATYPE):

    df_train = pd.read_csv(train_file, header=None)
    df_test = pd.read_csv(test_file, header=None)

    Xy_train = df_train.values.astype(dtype)
    Xy_test = df_test.values.astype(dtype)
    
    seqlen = Xy_train.shape[1]

    df_y_train = Xy_train[:,0].astype('int')
    df_y_test = Xy_test[:,0].astype('int')

    Y_train = np_utils.to_categorical(df_y_train, classes)
    Y_test = np_utils.to_categorical(df_y_test, classes)

    X_train = Xy_train[:, 1:seqlen].astype(dtype)
    X_test = Xy_test[:, 1:seqlen].astype(dtype)
    
    mat = np.concatenate((X_train, X_test), axis=0)
    # Scale data
    if scaling is not None:
        mat = scale_array(mat, scaling)

    # Separate training in training and validation splits after scaling
    X_train = mat[:X_train.shape[0], :]
    X_test = mat[X_train.shape[0]:, :]

    return X_train, Y_train, X_test, Y_test


