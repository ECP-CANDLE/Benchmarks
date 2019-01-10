from __future__ import absolute_import

import numpy as np
import pandas as pd

from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler

from keras.utils import np_utils

from default_utils import DEFAULT_SEED
from default_utils import DEFAULT_DATATYPE

def convert_to_class(y_one_hot, dtype=int):
    """ Converts a one-hot class encoding (array with as many positions as total
        classes, with 1 in the corresponding class position, 0 in the other positions),
        or soft-max class encoding (array with as many positions as total
        classes, whose largest valued position is used as class membership)
        to an integer class encoding.

        Parameters
        ----------
        y_one_hot : numpy array
            Input array with one-hot or soft-max class encoding.
        dtype : data type
            Data type to use for the output numpy array.
            (Default: int, integer data is used to represent the
            class membership).

        Return
        ----------
        Returns a numpy array with an integer class encoding.
    """

    maxi = lambda a: a.argmax()
    iter_to_na = lambda i: np.fromiter(i, dtype=dtype)
    return np.array([maxi(a) for a in y_one_hot])


def scale_array(mat, scaling=None):
    """ Scale data included in numpy array.
        
        Parameters
        ----------
        mat : numpy array
            Array to scale
        scaling : string
            String describing type of scaling to apply.
            Options recognized: 'maxabs', 'minmax', 'std'.
            'maxabs' : scales data to range [-1 to 1].
            'minmax' : scales data to range [-1 to 1].
            'std'    : scales data to normal variable with mean 0 and standard deviation 1.
            (Default: None, no scaling).

        Return
        ----------
        Returns the numpy array scaled by the method specified. \
        If no scaling method is specified, it returns the numpy \
        array unmodified.
    """
    
    if scaling is None or scaling.lower() == 'none':
        return mat

    # Scaling data
    if scaling == 'maxabs':
        # Scaling to [-1, 1]
        scaler = MaxAbsScaler(copy=False)
    elif scaling == 'minmax':
        # Scaling to [0,1]
        scaler = MinMaxScaler(copy=False)
    else:
        # Standard normalization
        scaler = StandardScaler(copy=False)
    
    return scaler.fit_transform(mat)



def impute_and_scale_array(mat, scaling=None):
    """ Impute missing values with mean and scale data included in numpy array.

        Parameters
        ----------
        mat : numpy array
            Array to scale
        scaling : string
            String describing type of scaling to apply.
            Options recognized: 'maxabs', 'minmax', 'std'.
            'maxabs' : scales data to range [-1 to 1].
            'minmax' : scales data to range [-1 to 1].
            'std'    : scales data to normal variable with mean 0 and standard deviation 1.
            (Default: None, no scaling).

        Return
        ----------
        Returns the numpy array imputed with the mean value of the \
        column and scaled by the method specified. If no scaling method is specified, \
        it returns the imputed numpy array.
    """
    
    imputer = Imputer(strategy='mean', axis=0, copy=False)
    imputer.fit_transform(mat)
    #mat = imputer.fit_transform(mat)
    
    return scale_array(mat, scaling)



def load_X_data(train_file, test_file,
                drop_cols=None, n_cols=None, shuffle=False, scaling=None,
                dtype=DEFAULT_DATATYPE, seed=DEFAULT_SEED):
    """ Load training and testing unlabeleled data from the files specified
        and construct corresponding training and testing pandas DataFrames.
        Columns to load can be selected or dropped. Order of rows
        can be shuffled. Data can be rescaled.
        Training and testing partitions (coming from the respective files)
        are preserved.
        This function assumes that the files contain a header with column names.

        Parameters
        ----------
        train_file : filename
            Name of the file to load the training data.
        test_file : filename
            Name of the file to load the testing data.
        drop_cols : list
            List of column names to drop from the files being loaded.
            (Default: None, all the columns are used).
        n_cols : integer
            Number of columns to load from the files.
            (Default: None, all the columns are used).
        shuffle : boolean
            Boolean flag to indicate row shuffling. If True the rows are
            re-ordered, if False the order in which rows are read is
            preserved.
            (Default: False, no permutation of the loading row order).
        scaling : string
            String describing type of scaling to apply.
            Options recognized: 'maxabs', 'minmax', 'std'.
            'maxabs' : scales data to range [-1 to 1].
            'minmax' : scales data to range [-1 to 1].
            'std'    : scales data to normal variable with \
                       mean 0 and standard deviation 1.
            (Default: None, no scaling).
        dtype : data type
            Data type to use for the output pandas DataFrames.
            (Default: DEFAULT_DATATYPE defined in default_utils).
        seed : int
            Value to intialize or re-seed the generator.
            (Default: DEFAULT_SEED defined in default_utils).

        Return
        ----------
        X_train : pandas DataFrame
            Data for training loaded in a pandas DataFrame and pre-processed as specified.
        X_test : pandas DataFrame
            Data for testing loaded in a pandas DataFrame and pre-processed as specified.
    """

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
    """ Load training and testing unlabeleled data from the files specified.
        Further split trainig data into training and validation partitions,
        and construct corresponding training, validation and testing pandas DataFrames.
        Columns to load can be selected or dropped. Order of rows
        can be shuffled. Data can be rescaled.
        Training and testing partitions (coming from the respective files)
        are preserved, but training is split into training and validation partitions.
        This function assumes that the files contain a header with column names.

        Parameters
        ----------
        train_file : filename
            Name of the file to load the training data.
        test_file : filename
            Name of the file to load the testing data.
        drop_cols : list
            List of column names to drop from the files being loaded.
            (Default: None, all the columns are used).
        n_cols : integer
            Number of columns to load from the files.
            (Default: None, all the columns are used).
        shuffle : boolean
            Boolean flag to indicate row shuffling. If True the rows are
            re-ordered, if False the order in which rows are read is
            preserved.
            (Default: False, no permutation of the loading row order).
        scaling : string
            String describing type of scaling to apply.
            Options recognized: 'maxabs', 'minmax', 'std'.
            'maxabs' : scales data to range [-1 to 1].
            'minmax' : scales data to range [-1 to 1].
            'std'    : scales data to normal variable with mean 0 and standard deviation 1.
            (Default: None, no scaling).
        validation_split : float
            Fraction of training data to set aside for validation.
            (Default: 0.1, ten percent of the training data is
            used for the validation partition).
        dtype : data type
            Data type to use for the output pandas DataFrames.
            (Default: DEFAULT_DATATYPE defined in default_utils).
        seed : int
            Value to intialize or re-seed the generator.
            (Default: DEFAULT_SEED defined in default_utils).


        Return
        ----------
        X_train : pandas DataFrame
            Data for training loaded in a pandas DataFrame and
            pre-processed as specified.
        X_val : pandas DataFrame
            Data for validation loaded in a pandas DataFrame and
            pre-processed as specified.
        X_test : pandas DataFrame
            Data for testing loaded in a pandas DataFrame and
            pre-processed as specified.
    """

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
    """ Load training and testing data from the files specified, with a column indicated to use as label.
        Construct corresponding training and testing pandas DataFrames,
        separated into data (i.e. features) and labels. Labels to output are one-hot encoded (categorical).
        Columns to load can be selected or dropped. Order of rows
        can be shuffled. Data can be rescaled.
        Training and testing partitions (coming from the respective files)
        are preserved.
        This function assumes that the files contain a header with column names.

        Parameters
        ----------
        train_file : filename
            Name of the file to load the training data.
        test_file : filename
            Name of the file to load the testing data.
        class_col : integer
            Index of the column to use as the label.
            (Default: None, this would cause the function to fail, a label
            has to be indicated at calling).
        drop_cols : list
            List of column names to drop from the files being loaded.
            (Default: None, all the columns are used).
        n_cols : integer
            Number of columns to load from the files.
            (Default: None, all the columns are used).
        shuffle : boolean
            Boolean flag to indicate row shuffling. If True the rows are
            re-ordered, if False the order in which rows are read is
            preserved.
            (Default: False, no permutation of the loading row order).
        scaling : string
            String describing type of scaling to apply.
            Options recognized: 'maxabs', 'minmax', 'std'.
            'maxabs' : scales data to range [-1 to 1].
            'minmax' : scales data to range [-1 to 1].
            'std'    : scales data to normal variable with mean 0 and standard deviation 1.
            (Default: None, no scaling).
        dtype : data type
            Data type to use for the output pandas DataFrames.
            (Default: DEFAULT_DATATYPE defined in default_utils).
        seed : int
            Value to intialize or re-seed the generator.
            (Default: DEFAULT_SEED defined in default_utils).


        Return
        ----------
        X_train : pandas DataFrame
            Data features for training loaded in a pandas DataFrame and
            pre-processed as specified.
        y_train : pandas DataFrame
            Data labels for training loaded in a pandas DataFrame.
            One-hot encoding (categorical) is used.
        X_test : pandas DataFrame
            Data features for testing loaded in a pandas DataFrame and
            pre-processed as specified.
        y_test : pandas DataFrame
            Data labels for testing loaded in a pandas DataFrame.
            One-hot encoding (categorical) is used.
    """

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
    """ Load training and testing data from the files specified, with a column indicated to use as label.
        Further split trainig data into training and validation partitions,
        and construct corresponding training, validation and testing pandas DataFrames,
        separated into data (i.e. features) and labels. Labels to output are one-hot encoded (categorical).
        Columns to load can be selected or dropped. Order of rows
        can be shuffled. Data can be rescaled.
        Training and testing partitions (coming from the respective files)
        are preserved, but training is split into training and validation partitions.
        This function assumes that the files contain a header with column names.

        Parameters
        ----------
        train_file : filename
            Name of the file to load the training data.
        test_file : filename
            Name of the file to load the testing data.
        class_col : integer
            Index of the column to use as the label.
            (Default: None, this would cause the function to fail, a label
            has to be indicated at calling).
        drop_cols : list
            List of column names to drop from the files being loaded.
            (Default: None, all the columns are used).
        n_cols : integer
            Number of columns to load from the files.
            (Default: None, all the columns are used).
        shuffle : boolean
            Boolean flag to indicate row shuffling. If True the rows are
            re-ordered, if False the order in which rows are loaded is
            preserved.
            (Default: False, no permutation of the loading row order).
        scaling : string
            String describing type of scaling to apply.
            Options recognized: 'maxabs', 'minmax', 'std'.
            'maxabs' : scales data to range [-1 to 1].
            'minmax' : scales data to range [-1 to 1].
            'std'    : scales data to normal variable with mean 0 and standard deviation 1.
            (Default: None, no scaling).
        validation_split : float
            Fraction of training data to set aside for validation.
            (Default: 0.1, ten percent of the training data is
            used for the validation partition).
        dtype : data type
            Data type to use for the output pandas DataFrames.
            (Default: DEFAULT_DATATYPE defined in default_utils).
        seed : int
            Value to intialize or re-seed the generator.
            (Default: DEFAULT_SEED defined in default_utils).


        Return
        ----------
        X_train : pandas DataFrame
            Data features for training loaded in a pandas DataFrame and
            pre-processed as specified.
        y_train : pandas DataFrame
            Data labels for training loaded in a pandas DataFrame.
            One-hot encoding (categorical) is used.
        X_val : pandas DataFrame
            Data features for validation loaded in a pandas DataFrame and
            pre-processed as specified.
        y_val : pandas DataFrame
            Data labels for validation loaded in a pandas DataFrame.
            One-hot encoding (categorical) is used.
        X_test : pandas DataFrame
            Data features for testing loaded in a pandas DataFrame and
            pre-processed as specified.
        y_test : pandas DataFrame
            Data labels for testing loaded in a pandas DataFrame.
            One-hot encoding (categorical) is used.
    """

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
    """ Load training and testing data from the files specified, with a column indicated to use as label.
        Further split trainig data into training and validation partitions,
        and construct corresponding training, validation and testing pandas DataFrames,
        separated into data (i.e. features) and labels.
        Labels to output can be integer labels (for classification) or
        continuous labels (for regression).
        Columns to load can be selected or dropped. Order of rows
        can be shuffled. Data can be rescaled.
        Training and testing partitions (coming from the respective files)
        are preserved, but training is split into training and validation partitions.
        This function assumes that the files contain a header with column names.

        Parameters
        ----------
        train_file : filename
            Name of the file to load the training data.
        test_file : filename
            Name of the file to load the testing data.
        class_col : integer
            Index of the column to use as the label.
            (Default: None, this would cause the function to fail, a label
            has to be indicated at calling).
        drop_cols : list
            List of column names to drop from the files being loaded.
            (Default: None, all the columns are used).
        n_cols : integer
            Number of columns to load from the files.
            (Default: None, all the columns are used).
        shuffle : boolean
            Boolean flag to indicate row shuffling. If True the rows are
            re-ordered, if False the order in which rows are loaded is
            preserved.
            (Default: False, no permutation of the loading row order).
        scaling : string
            String describing type of scaling to apply.
            Options recognized: 'maxabs', 'minmax', 'std'.
            'maxabs' : scales data to range [-1 to 1].
            'minmax' : scales data to range [-1 to 1].
            'std'    : scales data to normal variable with mean 0 and standard deviation 1.
            (Default: None, no scaling).
        validation_split : float
            Fraction of training data to set aside for validation.
            (Default: 0.1, ten percent of the training data is
            used for the validation partition).
        dtype : data type
            Data type to use for the output pandas DataFrames.
            (Default: DEFAULT_DATATYPE defined in default_utils).
        seed : int
            Value to intialize or re-seed the generator.
            (Default: DEFAULT_SEED defined in default_utils).


        Return
        ----------
        X_train : pandas DataFrame
            Data features for training loaded in a pandas DataFrame and
            pre-processed as specified.
        y_train : pandas DataFrame
            Data labels for training loaded in a pandas DataFrame.
        X_val : pandas DataFrame
            Data features for validation loaded in a pandas DataFrame and
            pre-processed as specified.
        y_val : pandas DataFrame
            Data labels for validation loaded in a pandas DataFrame.
        X_test : pandas DataFrame
            Data features for testing loaded in a pandas DataFrame and
            pre-processed as specified.
        y_test : pandas DataFrame
            Data labels for testing loaded in a pandas DataFrame.
    """

    assert class_col != None
    
    (X_train, y_train_oh), (X_val, y_val_oh), (X_test, y_test_oh) = load_Xy_one_hot_data2(train_file, test_file,
                                                                                 class_col, drop_cols, n_cols, shuffle, scaling,
                                                                                 validation_split, dtype, seed)

    y_train = convert_to_class(y_train_oh)
    y_val = convert_to_class(y_val_oh)
    y_test = convert_to_class(y_test_oh)
    

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def load_Xy_data_noheader(train_file, test_file, classes, usecols=None, scaling=None, dtype=DEFAULT_DATATYPE):
    """ Load training and testing data from the files specified, with the first column to use as label.
        Construct corresponding training and testing pandas DataFrames,
        separated into data (i.e. features) and labels.
        Labels to output are one-hot encoded (categorical).
        Columns to load can be selected. Data can be rescaled.
        Training and testing partitions (coming from the respective files)
        are preserved.
        This function assumes that the files do not contain a header.

        Parameters
        ----------
        train_file : filename
            Name of the file to load the training data.
        test_file : filename
            Name of the file to load the testing data.
        classes : integer
            Number of total classes to consider when
            building the categorical (one-hot) label encoding.
        usecols : list
            List of column indices to load from the files.
            (Default: None, all the columns are used).
        scaling : string
            String describing type of scaling to apply.
            Options recognized: 'maxabs', 'minmax', 'std'.
            'maxabs' : scales data to range [-1 to 1].
            'minmax' : scales data to range [-1 to 1].
            'std'    : scales data to normal variable with mean 0 and standard deviation 1.
            (Default: None, no scaling).
        dtype : data type
            Data type to use for the output pandas DataFrames.
            (Default: DEFAULT_DATATYPE defined in default_utils).

        Return
        ----------
        X_train : pandas DataFrame
            Data features for training loaded in a pandas DataFrame and
            pre-processed as specified.
        Y_train : pandas DataFrame
            Data labels for training loaded in a pandas DataFrame.
            One-hot encoding (categorical) is used.
        X_test : pandas DataFrame
            Data features for testing loaded in a pandas DataFrame and
            pre-processed as specified.
        Y_test : pandas DataFrame
            Data labels for testing loaded in a pandas DataFrame.
            One-hot encoding (categorical) is used.
    """
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

    mat = np.concatenate((X_train, X_test), axis=0)
    # Scale data
    if scaling is not None:
        mat = scale_array(mat, scaling)

    X_train = mat[:X_train.shape[0], :]
    X_test = mat[X_train.shape[0]:, :]

    return X_train, Y_train, X_test, Y_test


def load_csv_data(train_path, test_path=None, sep=',', nrows=None,
                  x_cols=None, y_cols=None, drop_cols=None,
                  onehot_cols=None, n_cols=None, random_cols=False,
                  shuffle=False, scaling=None, dtype=None,
                  validation_split=None, return_dataframe=True,
                  return_header=False, seed=DEFAULT_SEED):

    """ Load data from the files specified.
        Columns corresponding to data features and labels can be specified. A one-hot
        encoding can be used for either features or labels.
        If validation_split is specified, trainig data is further split into training
        and validation partitions.
        pandas DataFrames are used to load and pre-process the data. If specified,
        those DataFrames are returned. Otherwise just values are returned.
        Labels to output can be integer labels (for classification) or
        continuous labels (for regression).
        Columns to load can be specified, randomly selected or a subset can be dropped.
        Order of rows can be shuffled. Data can be rescaled.
        This function assumes that the files contain a header with column names.

        Parameters
        ----------
        train_path : filename
            Name of the file to load the training data.
        test_path : filename
            Name of the file to load the testing data. (Optional).
        sep : character
            Character used as column separator.
            (Default: ',', comma separated values).
        nrows : integer
            Number of rows to load from the files.
            (Default: None, all the rows are used).
        x_cols : list
            List of columns to use as features.
            (Default: None).
        y_cols : list
            List of columns to use as labels.
            (Default: None).
        drop_cols : list
            List of columns to drop from the files being loaded.
            (Default: None, all the columns are used).
        onehot_cols : list
            List of columns to one-hot encode.
            (Default: None).
        n_cols : integer
            Number of columns to load from the files.
            (Default: None).
        random_cols : boolean
            Boolean flag to indicate random selection of columns.
            If True a number of n_cols columns is randomly selected, if False
            the specified columns are used.
            (Default: False).
        shuffle : boolean
            Boolean flag to indicate row shuffling. If True the rows are
            re-ordered, if False the order in which rows are read is
            preserved.
            (Default: False, no permutation of the loading row order).
        scaling : string
            String describing type of scaling to apply.
            Options recognized: 'maxabs', 'minmax', 'std'.
            'maxabs' : scales data to range [-1 to 1].
            'minmax' : scales data to range [-1 to 1].
            'std'    : scales data to normal variable with mean 0 and standard deviation 1.
            (Default: None, no scaling).
        dtype : data type
            Data type to use for the output pandas DataFrames.
            (Default: None).
        validation_split : float
            Fraction of training data to set aside for validation.
            (Default: None, no validation partition is constructed).
        return_dataframe : boolean
            Boolean flag to indicate that the pandas DataFrames
            used for data pre-processing are to be returned.
            (Default: True, pandas DataFrames are returned).
        return_header : boolean
            Boolean flag to indicate if the column headers are
            to be returned.
            (Default: False, no column headers are separetely returned).
        seed : int
            Value to intialize or re-seed the generator.
            (Default: DEFAULT_SEED defined in default_utils).


        Return
        ----------
        Tuples of data features and labels are returned, for \
        train, validation and testing partitions, together with the column \
        names (headers). The specific objects to return depend \
        on the options selected.
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

