import sys
import pandas as pd
import numpy as np
from scipy import stats
from collections import Counter


def quantile_normalization(data):
    '''
    This function does quantile normalization to input data. After normalization, the samples (rows) in output
    data follow the same distribution, which is the average distribution calculated based on all samples.
    This function allows missing values, and assume missing values occur at random.

    Parameters:
    -----------
    data: numpy array or pandas data frame of numeric values, with a shape of [n_samples, n_features].

    Returns:
    --------
    norm_data: numpy array or pandas data frame containing the data after quantile normalization.
    '''

    colnames = None
    rownames = None
    if isinstance(data, pd.DataFrame):
        colnames = data.columns
        rownames = data.index
        data = data.values
    elif not isinstance(data, np.ndarray):
        print('Input data must be a numpy array or pandas data frame')
        sys.exit(1)

    norm_data = data.copy()
    nan_mask = np.isnan(norm_data)
    if np.sum(nan_mask) > 0:
        n_samples, n_features = norm_data.shape
        for i in range(n_samples):
            idi_nan = np.where(np.isnan(norm_data[i, :]))[0]
            if len(idi_nan) > 0:
                idi = np.setdiff1d(range(n_features), idi_nan)
                norm_data[i, idi_nan] = np.random.choice(norm_data[i, idi], size=len(idi_nan), replace=True)

    quantiles = np.mean(np.sort(norm_data, axis=1), axis=0)
    ranks = np.apply_along_axis(stats.rankdata, 1, norm_data)
    rank_indices = ranks.astype(int) - 1
    norm_data = quantiles[rank_indices]

    if np.sum(nan_mask) > 0:
        row_id, col_id = np.where(nan_mask)
        norm_data[row_id, col_id] = np.nan

    if colnames is not None and rownames is not None:
        norm_data = pd.DataFrame(norm_data, columns=colnames, index=rownames)

    return norm_data


def generate_cross_validation_partition(group_label, n_folds=5, n_repeats=1, portions=None, random_seed=None):
    '''
    This function generates partition indices of samples for cross-validation analysis.

    Parameters:
    -----------
    group_label: 1-D array or list of group labels of samples. If there are no groups in samples, a list of
        sample indices can be supplied for generating partitions based on individual samples rather than sample groups.
    n_folds: positive integer larger than 1, indicating the number of folds for cross-validation. Default is 5.
    n_repeats: positive integer, indicating how many times the n_folds cross-validation should be repeated.
        So the total number of cross-validation trials is n_folds * n_repeats. Default is 1.
    portions: 1-D array or list of positive integers, indicating the number of data folds in each set
        (e.g. training set, testing set, or validation set) after partitioning. The summation of elements
        in portions must be equal to n_folds. Default is [1, n_folds - 1].
    random_seed: positive integer, the seed for random generator. Default is None.

    Returns:
    --------
    partition: list of n_folds * n_repeats lists, each of which contains len(portions) sample index lists for
        a cross-validation trial.
    '''

    group_counter = Counter(group_label)
    unique_label = np.array(list(group_counter.keys()))
    n_group = len(unique_label)
    if n_group < n_folds:
        print('The number of groups in labels can not be smaller than the number of folds.')
        sys.exit(1)
    sorted_label = np.array(sorted(unique_label, key=lambda x: group_counter[x], reverse=True))

    if portions is None:
        portions = [1, n_folds - 1]
    else:
        if np.sum(portions) != n_folds:
            print('The summation of elements in portions must be equal to n_folds')
            sys.exit(1)

    if random_seed is not None:
        np.random.seed(random_seed)

    n_set = len(portions)
    partition = []
    for r in range(n_repeats):

        if r == 0 and random_seed is None:
            label = sorted_label.copy()
        else:
            idr = np.random.permutation(n_group)
            label = sorted_label[idr]

        folds = [[] for _ in range(n_folds)]
        fold_size = np.zeros((n_folds, ))

        for g in range(n_group):
            f = np.argmin(fold_size)
            folds[f].append(label[g])
            fold_size[f] += group_counter[label[g]]

        for f in range(n_folds):
            folds[f] = list(np.where(np.isin(group_label, folds[f]))[0])

        a = list(range(n_folds)) + list(range(n_folds))
        for f in range(n_folds):
            temp = []
            end = f
            for s in range(n_set):
                start = end
                end = start + portions[s]
                t = []
                for i in range(start, end):
                    t = t + folds[a[i]]
                temp.append(sorted(t))
            partition.append(temp)

    return partition
