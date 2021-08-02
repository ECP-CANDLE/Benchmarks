import sys
import pandas as pd
import numpy as np
import numpy.linalg as la
from astropy.stats import median_absolute_deviation
import matplotlib.pyplot as plt


def select_features_by_missing_values(data, threshold=0.1):
    '''
    This function returns the indices of the features whose missing rates are smaller than the threshold.

    Parameters:
    -----------
    data: numpy array or pandas data frame of numeric values, with a shape of [n_samples, n_features]
    threshold: float in the range of [0, 1]. Features with a missing rate smaller than threshold will be selected.
        Default is 0.1

    Returns:
    --------
    indices: 1-D numpy array containing the indices of selected features
    '''

    if isinstance(data, pd.DataFrame):
        data = data.values
    elif not isinstance(data, np.ndarray):
        print('Input data must be a numpy array or pandas data frame')
        sys.exit(1)

    missing_rate = np.sum(np.isnan(data), axis=0) / data.shape[0]
    indices = np.where(missing_rate < threshold)[0]

    indices = np.sort(indices)

    return indices


def select_features_by_variation(data, variation_measure='var', threshold=None, portion=None, draw_histogram=False,
                                 bins=100, log=False):
    '''
    This function evaluates the variations of individual features and returns the indices of features with large
    variations. Missing values are ignored in evaluating variation.

    Parameters:
    -----------
    data: numpy array or pandas data frame of numeric values, with a shape of [n_samples, n_features].
    variation_metric: string indicating the metric used for evaluating feature variation. 'var' indicates variance;
        'std' indicates standard deviation; 'mad' indicates median absolute deviation. Default is 'var'.
    threshold: float. Features with a variation larger than threshold will be selected. Default is None.
    portion: float in the range of [0, 1]. It is the portion of features to be selected based on variation.
        The number of selected features will be the smaller of int(portion * n_features) and the total number of
        features with non-missing variations. Default is None. threshold and portion can not take real values
        and be used simultaneously.
    draw_histogram: boolean, whether to draw a histogram of feature variations. Default is False.
    bins: positive integer, the number of bins in the histogram. Default is the smaller of 50 and the number of
        features with non-missing variations.
    log: boolean, indicating whether the histogram should be drawn on log scale.


    Returns:
    --------
    indices: 1-D numpy array containing the indices of selected features. If both threshold and
        portion are None, indices will be an empty array.
    '''

    if isinstance(data, pd.DataFrame):
        data = data.values
    elif not isinstance(data, np.ndarray):
        print('Input data must be a numpy array or pandas data frame')
        sys.exit(1)

    if variation_measure == 'std':
        v_all = np.nanstd(a=data, axis=0)
    elif variation_measure == 'mad':
        v_all = median_absolute_deviation(data=data, axis=0, ignore_nan=True)
    else:
        v_all = np.nanvar(a=data, axis=0)

    indices = np.where(np.invert(np.isnan(v_all)))[0]
    v = v_all[indices]

    if draw_histogram:
        if len(v) < 50:
            print('There must be at least 50 features with variation measures to draw a histogram')
        else:
            bins = int(min(bins, len(v)))
            _ = plt.hist(v, bins=bins, log=log)
            plt.show()

    if threshold is None and portion is None:
        return np.array([])
    elif threshold is not None and portion is not None:
        print('threshold and portion can not be used simultaneously. Only one of them can take a real value')
        sys.exit(1)

    if threshold is not None:
        indices = indices[np.where(v > threshold)[0]]
    else:
        n_f = int(min(portion * data.shape[1], len(v)))
        indices = indices[np.argsort(-v)[:n_f]]

    indices = np.sort(indices)

    return indices


def select_decorrelated_features(data, method='pearson', threshold=None, random_seed=None):
    '''
    This function selects features whose mutual absolute correlation coefficients are smaller than a threshold.
    It allows missing values in data. The correlation coefficient of two features are calculated based on
    the observations that are not missing in both features. Features with only one or no value present and
    features with a zero standard deviation are not considered for selection.

    Parameters:
    -----------
    data: numpy array or pandas data frame of numeric values, with a shape of [n_samples, n_features].
    method: string indicating the method used for calculating correlation coefficient. 'pearson' indicates Pearson
        correlation coefficient; 'kendall' indicates Kendall Tau correlation coefficient; 'spearman' indicates
        Spearman rank correlation coefficient. Default is 'pearson'.
    threshold: float. If two features have an absolute correlation coefficient higher than threshold,
        one of the features is removed. If threshold is None, a feature is removed only when the two features
        are exactly identical. Default is None.
    random_seed: positive integer, seed of random generator for ordering the features. If it is None, features
        are not re-ordered before feature selection and thus the first feature is always selected. Default is None.

    Returns:
    --------
    indices: 1-D numpy array containing the indices of selected features.
    '''

    if isinstance(data, np.ndarray):
        data = pd.DataFrame(data)
    elif not isinstance(data, pd.DataFrame):
        print('Input data must be a numpy array or pandas data frame')
        sys.exit(1)

    present = np.where(np.sum(np.invert(pd.isna(data)), axis=0) > 1)[0]
    present = present[np.where(np.nanstd(data.iloc[:, present].values, axis=0) > 0)[0]]

    data = data.iloc[:, present]

    num_f = data.shape[1]
    if random_seed is not None:
        np.random.seed(random_seed)
        random_order = np.random.permutation(num_f)
        data = data.iloc[:, random_order]

    if threshold is not None:
        if np.sum(pd.isna(data).values) == 0 and method == 'pearson':
            cor = np.corrcoef(data.values, rowvar=False)
        else:
            cor = data.corr(method=method).values
    else:
        data = data.values

    rm = np.full(num_f, False)
    index = 0
    while index < num_f - 1:
        if rm[index]:
            index += 1
            continue
        idi = np.array(range(index + 1, num_f))
        idi = idi[np.where(rm[idi] == False)[0]]
        if len(idi) > 0:
            if threshold is None:
                idi = idi[np.where(np.sum(np.isnan(data[:, idi]) ^ np.isnan(data[:, index][:, np.newaxis]), axis=0) == 0)[0]]
                if len(idi) > 0:
                    idi = idi[np.where(np.nansum(abs(data[:, idi] - data[:, index][:, np.newaxis]), axis=0) == 0)[0]]
            else:
                idi = idi[np.where(abs(cor[index, idi]) >= threshold)[0]]
            if len(idi) > 0:
                rm[idi] = True
        index += 1

    indices = np.where(rm == False)[0]
    if random_seed is not None:
        indices = random_order[indices]
    indices = np.sort(present[indices])

    return indices
