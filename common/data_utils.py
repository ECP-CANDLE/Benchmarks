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


def load_Xy_data_noheader(train_file, test_file, classes, usecols=None, scaling=None, dtype=DEFAULT_DATATYPE):

    print('Loading data...')
    df_train = (pd.read_csv(train_file, header=None, usecols=usecols).values).astype(dtype)
    df_test = (pd.read_csv(test_file, header=None, usecols=usecols).values).astype(dtype)
    print('done')

    print('df_train shape:', df_train.shape)
    print('df_test shape:', df_test.shape)

    seqlen = df_train.shape[1]

    df_y_train = df_train[:,0].astype('int')
    df_y_test = df_test[:,0].astype('int')

    Y_train = np_utils.to_categorical(df_y_train, classes)
    Y_test = np_utils.to_categorical(df_y_test, classes)

    df_x_train = df_train[:, 1:seqlen].astype(dtype)
    df_x_test = df_test[:, 1:seqlen].astype(dtype)

#        X_train = df_x_train.as_matrix()
#        X_test = df_x_test.as_matrix()

    X_train = df_x_train
    X_test = df_x_test

    scaler = MaxAbsScaler()
    mat = np.concatenate((X_train, X_test), axis=0)
    # Scale data
    if scaling is not None:
        mat = scaler.fit_transform(mat)

    X_train = mat[:X_train.shape[0], :]
    X_test = mat[X_train.shape[0]:, :]

    return X_train, Y_train, X_test, Y_test


def load_csv_data(train_path, test_path=None, sep=',', nrows=None,
                  x_cols=None, y_cols=None, drop_cols=None,
                  onehot_cols=None, n_cols=None, random_cols=False,
                  shuffle=False, scaling=None, dtype=None,
                  validation_split=None, return_dataframe=True,
                  return_header=False, seed=2017):

    """Load training and test data from CSV

        Parameters
        ----------
        train_path : path
            training file path
    """

    if x_cols is None and drop_cols is None and n_cols is None:
        usecols = None
        y_names = None
    else:
        df_cols = pd.read_csv(train_path, engine='c', sep=sep, nrows=0)
        df_x_cols = df_cols.copy()
        # drop columns by name or index
        if y_cols is not None:
            df_x_cols = df_x_cols.drop(df_cols[y_cols], axis=1)
        if drop_cols is not None:
            df_x_cols = df_x_cols.drop(df_cols[drop_cols], axis=1)

        reserved = []
        if onehot_cols is not None:
            reserved += onehot_cols
        if x_cols is not None:
            reserved += x_cols

        nx = df_x_cols.shape[1]
        if n_cols and n_cols < nx:
            if random_cols:
                indexes = sorted(np.random.choice(list(range(nx)), n_cols, replace=False))
            else:
                indexes = list(range(n_cols))
            x_names = list(df_x_cols[indexes])
            unreserved = [x for x in x_names if x not in reserved]
            n_keep = np.maximum(n_cols - len(reserved), 0)
            combined = reserved + unreserved[:n_keep]
            x_names = [x for x in df_x_cols if x in combined]
        elif x_cols is not None:
            x_names = list(df_x_cols[x_cols])
        else:
            x_names = list(df_x_cols.columns)

        usecols = x_names
        if y_cols is not None:
            y_names = list(df_cols[y_cols])
            usecols = y_names + x_names

    df_train = pd.read_csv(train_path, engine='c', sep=sep, nrows=nrows, usecols=usecols)
    if test_path:
        df_test = pd.read_csv(test_path, engine='c', sep=sep, nrows=nrows, usecols=usecols)
    else:
        df_test = df_train[0:0].copy()

    if y_cols is None:
        y_names = []
    elif y_names is None:
        y_names = list(df_train[0:0][y_cols])

    if shuffle:
        df_train = df_train.sample(frac=1, random_state=seed)
        if test_path:
            df_test = df_test.sample(frac=1, random_state=seed)

    df_cat = pd.concat([df_train, df_test])
    df_y = df_cat[y_names]
    df_x = df_cat.drop(y_names, axis=1)

    if onehot_cols is not None:
        for col in onehot_cols:
            if col in y_names:
                df_dummy = pd.get_dummies(df_y[col], prefix=col, prefix_sep=':')
                df_y = pd.concat([df_dummy, df_y.drop(col, axis=1)], axis=1)
                # print(df_dummy.columns)
            else:
                df_dummy = pd.get_dummies(df_x[col], prefix=col, prefix_sep=':')
                df_x = pd.concat([df_dummy, df_x.drop(col, axis=1)], axis=1)

    if scaling is not None:
        mat = scale_array(df_x.values, scaling)
        df_x = pd.DataFrame(mat, index=df_x.index, columns=df_x.columns, dtype=dtype)

    n_train = df_train.shape[0]

    x_train = df_x[:n_train]
    y_train = df_y[:n_train]
    x_test = df_x[n_train:]
    y_test = df_y[n_train:]

    return_y = y_cols is not None
    return_val = validation_split and validation_split > 0 and validation_split < 1
    return_test = test_path

    if return_val:
        n_val = int(n_train * validation_split)
        x_val = x_train[-n_val:]
        y_val = y_train[-n_val:]
        x_train = x_train[:-n_val]
        y_train = y_train[:-n_val]

    ret = [x_train]
    ret = ret + [y_train] if return_y else ret
    ret = ret + [x_val] if return_val else ret
    ret = ret + [y_val] if return_y and return_val else ret
    ret = ret + [x_test] if return_test else ret
    ret = ret + [y_test] if return_y and return_test else ret

    if not return_dataframe:
        ret = [x.values for x in ret]

    if return_header:
        ret = ret + [df_x.columns.tolist(), df_y.columns.tolist()]

    return tuple(ret) if len(ret) > 1 else ret

