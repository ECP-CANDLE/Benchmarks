#! /usr/bin/env python

from __future__ import division, print_function

import pandas as pd
import sys
import os
import dill

import numpy as np

from keras import backend as K


file_path = os.path.dirname(os.path.realpath(__file__))
lib_path = os.path.abspath(os.path.join(file_path, './'))
sys.path.append(lib_path)

import candle

additional_definitions = [
{'name': 'uqmode',
    'type': str,
    'default': None,
    'choices': ['hom', 'het', 'qtl', 'contam'],
    'help': 'mode of UQ regression used: homoscedastic (hom), heteroscedastic (het) or quantile (qtl)'},
{'name': 'plot_steps',
    'type': candle.str2bool,
    'default': False,
    'help': 'plot step-by-step the computation of the empirical calibration by binning'},
{'name': 'results_filename',
    'type': str,
    'default': None,
    'help': 'file with uq inference results'},
{'name': 'cv',
    'type': int,
    'default': 10,
    'help': 'number of cross validations for calibration by interpolation'},
{'name': 'sigma',
    'type': float,
    'default': None,
    'help': 'Standard deviation of normal distribution in contamination model'},
]

required = [
    'uqmode',
    'results_filename'
]

def coverage_80p(y_test, y_pred, std_pred, y_pred_1d=None, y_pred_9d=None):
    """ Determine the fraction of the true data that falls
        into the 80p coverage of the model.
        For homoscedastic and heteroscedastic models the
        standard deviation prediction is used.
        For quantile model, if first and nineth deciles
        are available, these are used instead of computing
        a standard deviation based on Gaussian assumptions.
        
        Parameters
        ----------
        y_test : numpy array
            True (observed) values array.
        y_pred : numpy array
            Mean predictions made by the model.
        std_pred : numpy array
            Standard deviation predictions made by the model.
        y_pred_1d : numpy array
            First decile predictions made by qtl model.
        y_pred_9d : numpy array
            Nineth decile predictions made by qtl model.
    """

    if std_pred is None: # for qtl
        topLim = y_pred_9d
        botLim = y_pred_1d
    else: # for hom and het
        topLim = y_pred + 1.28 * std_pred
        botLim = y_pred - 1.28 * std_pred
    
    # greater than top
    count_gr = np.count_nonzero(np.clip(y_test - topLim, 0., None))
    # less than bottom
    count_ls = np.count_nonzero(np.clip(botLim - y_test, 0., None))
    
    count_out = count_gr + count_ls
    N_test = y_test.shape[0]
    frac_out = (float(count_out)/float(N_test))
    
    return (1. - frac_out)
    

class CalibrationApp(candle.Benchmark):

    def set_locals(self):
        """Functionality to set variables specific for the benchmark
        - required: set of required parameters for the benchmark.
        - additional_definitions: list of dictionaries describing the additional parameters for the
        benchmark.
        """

        if required is not None:
            self.required = set(required)
        if additional_definitions is not None:
            self.additional_definitions = additional_definitions



def initialize_parameters(default_model='calibration_default.txt'):

    # Build benchmark object
    calBmk = CalibrationApp(file_path, default_model, 'python',
                            prog='calibration_main', desc='script to compute empirical calibration for UQ regression')

    # config_file, rng_seed and save_path from standard
    # Finalize parameters
    gParameters = candle.finalize_parameters(calBmk)

    return gParameters



def run(params):
    candle.set_seed(params['rng_seed'])
    uqmode = params['uqmode'] # hom, het, qtl
    filename = params['results_filename']
    cv = params['cv']

    index_dp = filename.find('DR=')
    if index_dp == -1: # DR is not in filename
        print('No dropout rate found in filename')
        print('Using -1 to denote NA')
        dp_perc = -1
    else:
        if filename[index_dp + 6] == '.':
            dp = float(filename[index_dp+3:index_dp+3+3])
        else:
            dp = float(filename[index_dp+3:index_dp+3+4])

        print('Droput rate: ', dp)
        dp_perc = dp * 100.
    method = uqmode + ' - dropout ' + str(dp_perc) + '%'
    prefix = params['output_dir'] + '/' + uqmode + '_DR=' + str(dp_perc)

    df_data = pd.read_csv(filename, sep='\t')
    print('data read shape: ', df_data.shape)
    # compute statistics according to uqmode
    if uqmode == 'hom':
        if df_data.shape[1] < 9:
            print('Too few columns... Asumming that a summary ' \
              + '(and not individual realizations) has been ' \
              + 'given as input')
            Ytest, Ypred_mean, yerror, sigma, Ypred_std, pred_name = candle.compute_statistics_homoscedastic_summary(df_data)
        else: # all individual realizations
            Ytest, Ypred_mean, yerror, sigma, Ypred_std, pred_name = candle.compute_statistics_homoscedastic(df_data)
        cov80p = coverage_80p(Ytest, Ypred_mean, sigma)
    elif uqmode == 'het': # for heteroscedastic UQ
        Ytest, Ypred_mean, yerror, sigma, Ypred_std, pred_name = candle.compute_statistics_heteroscedastic(df_data)
        cov80p = coverage_80p(Ytest, Ypred_mean, sigma)
    elif uqmode == 'qtl': # for quantile UQ
        Ytest, Ypred_mean, yerror, sigma, Ypred_std, pred_name, Ypred_1d_mean, Ypred_9d_mean = candle.compute_statistics_quantile(df_data)
        cov80p = coverage_80p(Ytest, Ypred_mean, None, Ypred_1d_mean, Ypred_9d_mean)
        decile_list = ['5th', '1st', '9th']
        candle.plot_decile_predictions(Ypred_mean, Ypred_1d_mean, Ypred_9d_mean, decile_list, pred_name, prefix)
    elif uqmode == 'contam':
        Ytest, Ypred_mean, yerror, sigma_, Ypred_std, pred_name = candle.compute_statistics_homoscedastic(df_data)
        sigma_scalar = params['sigma']
        if sigma_scalar is None:
            raise Exception('ERROR ! No sigma ' \
            + 'specified for contamination model... Exiting')
        sigma = sigma_scalar * np.ones(Ytest.shape[0])
        cov80p = coverage_80p(Ytest, Ypred_mean, sigma)
        print('Coverage (80%): ', cov80p)
        candle.plot_density_observed_vs_predicted(Ytest, Ypred_mean, pred_name, prefix)
        candle.plot_2d_density_sigma_vs_error(sigma, yerror, method, prefix)
        
        mse = np.mean((Ytest - Ypred_mean)**2)
        mae = np.mean(np.abs(Ytest - Ypred_mean))
        print('Prediction error in testing')
        print('MSE: ', mse)
        print('MAE: ', mae)
        candle.plot_contamination(Ytest, Ypred_mean, sigma, pred_name=pred_name, figprefix=prefix)
        print('Since in contamination model std prediction is ' \
            + 'uniform for all samples, no point in ' \
            + 'calibrating... Finishing')
        return
    else:
        raise Exception('ERROR ! UQ mode specified ' \
            + 'for calibration: ' + uqmode + ' not ' \
            + 'implemented... Exiting')

    print('Coverage (80%) before calibration: ', cov80p)

    # Density / Histogram plots
    candle.plot_density_observed_vs_predicted(Ytest, Ypred_mean, pred_name, prefix)
    candle.plot_2d_density_sigma_vs_error(sigma, yerror, method, prefix)
    candle.plot_histogram_error_per_sigma(sigma, yerror, method, prefix)
    
    # shuffle data for calibration
    index_perm_total, pSigma_cal, pSigma_test, pMean_cal, pMean_test, true_cal, true_test = candle.split_data_for_empirical_calibration(Ytest, Ypred_mean, sigma)

    # Compute empirical calibration by smooth interpolation
    splineobj1, splineobj2 = candle.compute_empirical_calibration_interpolation(pSigma_cal, pMean_cal, true_cal, cv)
    error = np.abs(true_cal - pMean_cal)
    candle.plot_calibration_interpolation(pSigma_cal, error, splineobj1, splineobj2, method, prefix, params['plot_steps'])
    
    # Check prediction error
    eabs_pred = splineobj2(pSigma_test)
    cov80p = coverage_80p(true_test, pMean_test, eabs_pred)
    print('Coverage (80%) after calibration: ', cov80p)
    eabs_true = np.abs(true_test - pMean_test)
    mse = np.mean((eabs_true - eabs_pred)**2)
    mae = np.mean(np.abs(eabs_true - eabs_pred))
    print('Prediction error in testing calibration')
    print('MSE: ', mse)
    print('MAE: ', mae)

    # Use MAE as threshold of accuracy
    # Mark samples with predicted std > mae
    candle.plot_calibrated_std(true_test, pMean_test, eabs_pred, 2.*mae, pred_name, prefix)
    
    # store calibration
    fname = prefix + '_calibration_interpolation_spline.dkl'
    with open(fname, 'wb') as f:
        dill.dump(splineobj2, f)
        print('Calibration spline (interpolation) stored in file: ', fname)
    
    

def main():
    params = initialize_parameters()
    run(params)


if __name__ == '__main__':
    main()


