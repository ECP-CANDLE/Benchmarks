#! /usr/bin/env python

from __future__ import division, print_function

import pandas as pd
import sys
import os
import pickle
import dill

lib_path2 = os.path.abspath(os.path.join('..', '..', 'common'))
sys.path.append(lib_path2)

import candle_keras as candle


def read_file(path, filename):

    df_data = pd.read_csv(path + filename, sep='\t')
    print('data read shape: ', df_data.shape)

    return df_data

def main():
    
    if ( len ( sys.argv ) < 3 ) :
        sys.stderr.write ( "\nUsage:  calibration_HOM.py PATH FILENAME [PLOT_STEPS_FLAG]\n" )
        sys.stderr.write ("FILENAME: usually <model>_pred.tsv\n")
        sys.exit ( 0 )

    path = sys.argv [1]
    filename = sys.argv [2]
    
    try:
        steps = sys.argv [3]
    except IndexError:
        steps = False
    
    folder_out = './outUQ/'
    if folder_out and not os.path.exists(folder_out):
        os.makedirs(folder_out)

    method = 'Dropout'
    prefix = folder_out + 'homoscedastic_DR'
    
    df_data = read_file(path, filename)
    Ytest, Ypred_mean, yerror, sigma, Ypred_std, pred_name = candle.compute_statistics_homoscedastic(df_data)

    #plots
    candle.plot_density_observed_vs_predicted(Ytest, Ypred_mean, pred_name, prefix)
    candle.plot_2d_density_sigma_vs_error(sigma, yerror, method, prefix)
    candle.plot_histogram_error_per_sigma(sigma, yerror, method, prefix)
    
    # shuffle data for calibration
    index_perm_total, pSigma_cal, pSigma_test, pMean_cal, pMean_test, true_cal, true_test = candle.split_data_for_empirical_calibration(Ytest, Ypred_mean, sigma)

    # Compute empirical calibration
    bins = 60
    coverage_percentile = 95
    mean_sigma, min_sigma, max_sigma, error_thresholds, err_err, error_thresholds_smooth, sigma_start_index, sigma_end_index, s_interpolate = candle.compute_empirical_calibration(pSigma_cal, pMean_cal, true_cal, bins, coverage_percentile)

    candle.plot_calibration_and_errors(mean_sigma, sigma_start_index, sigma_end_index,
                                min_sigma, max_sigma,
                                error_thresholds,
                                error_thresholds_smooth,
                                err_err,
                                s_interpolate,
                                coverage_percentile, method, prefix, steps)


    # Use empirical calibration and automatic determined monotonic interval
    minL_sigma_auto = mean_sigma[sigma_start_index]
    maxL_sigma_auto = mean_sigma[sigma_end_index]
    index_sigma_range_test, xp_test, yp_test, eabs_red = candle.applying_calibration(pSigma_test, pMean_test, true_test, s_interpolate, minL_sigma_auto, maxL_sigma_auto)
    # Check sigma overprediction
    p_cov = coverage_percentile
    num_cal = pSigma_cal.shape[0]
    pYstd_perm_all = Ypred_std[index_perm_total]
    pYstd_test = pYstd_perm_all[num_cal:]
    pYstd_red = pYstd_test[index_sigma_range_test]
    candle.overprediction_check(yp_test, eabs_red)

    # storing calibration
    fname = prefix + '_calibration_limits.pkl'
    with open(fname, 'wb') as f:
        pickle.dump([minL_sigma_auto, maxL_sigma_auto], f, protocol=4)
        print('Calibration limits stored in file: ', fname)
    fname = prefix + '_calibration_spline.dkl'
    with open(fname, 'wb') as f:
#        pickle.dump(s_interpolate, f, protocol=pickle.HIGHEST_PROTOCOL)
        dill.dump(s_interpolate, f)
        print('Calibration spline stored in file: ', fname)


if __name__ == '__main__':
    main()


