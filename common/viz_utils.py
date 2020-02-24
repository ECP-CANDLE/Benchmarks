from pathlib import Path
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import numpy as np

def plot_history(out, history, metric='loss', title=None, width=8, height=6):
    title = title or 'model {}'.format(metric)
    val_metric = 'val_{}'.format(metric)
    plt.figure(figsize=(width, height))
    plt.plot(history.history[metric], marker='o')
    plt.plot(history.history[val_metric], marker='d')
    plt.title(title)
    plt.ylabel(metric)
    plt.xlabel('epoch')
    plt.legend(['train_{}'.format(metric), 'val_{}'.format(metric)], loc='upper center')
    png = '{}.plot.{}.png'.format(out, metric)
    plt.savefig(png, bbox_inches='tight')

def plot_scatter(data, classes, out, width=10, height=8):
    cmap = plt.cm.get_cmap('gist_rainbow')
    plt.figure(figsize=(width, height))
    plt.scatter(data[:, 0], data[:, 1], c=classes, cmap=cmap, lw=0.5, edgecolor='black', alpha=0.7)
    plt.colorbar()
    png = '{}.png'.format(out)
    plt.savefig(png, bbox_inches='tight')

def plot_error(y_true, y_pred, batch, file_ext, file_pre='output_dir', subsample=1000):
    if batch % 10:
        return

    total = len(y_true)
    if subsample and subsample < total:
        usecols = np.random.choice(total, size=subsample, replace=False)
        y_true = y_true[usecols]
        y_pred = y_pred[usecols]

    y_true = y_true * 100
    y_pred = y_pred * 100
    diffs = y_pred - y_true

    bins = np.linspace(-200, 200, 100)
    if batch == 0:
        y_shuf = np.random.permutation(y_true)
        plt.hist(y_shuf - y_true, bins, alpha=0.5, label='Random')

    #plt.hist(diffs, bins, alpha=0.35-batch/100., label='Epoch {}'.format(batch+1))
    plt.hist(diffs, bins, alpha=0.3, label='Epoch {}'.format(batch+1))
    plt.title("Histogram of errors in percentage growth")
    plt.legend(loc='upper right')
    plt.savefig(file_pre+'.histogram'+file_ext+'.b'+str(batch)+'.png')
    plt.close()

    # Plot measured vs. predicted values
    fig, ax = plt.subplots()
    plt.grid('on')
    ax.scatter(y_true, y_pred, color='red', s=10)
    ax.plot([y_true.min(), y_true.max()],
            [y_true.min(), y_true.max()], 'k--', lw=4)
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    plt.savefig(file_pre+'.diff'+file_ext+'.b'+str(batch)+'.png')
    plt.close()

###### UTILS for UQ / CALIBRATION VISUALIZATION

from matplotlib.colors import LogNorm

def plot_density_observed_vs_predicted(Ytest, Ypred, pred_name=None, figprefix=None):
    """Functionality to plot a 2D histogram of the distribution of observed (ground truth)
       values vs. predicted values. The plot generated is stored in a png file.

    Parameters
    ----------
    Ytest : numpy array
      Array with (true) observed values
    Ypred : numpy array
      Array with predicted values.
    pred_name : string
      Name of data colum or quantity predicted (e.g. growth, AUC, etc.)
    figprefix : string
      String to prefix the filename to store the figure generated.
      A '_density_predictions.png' string will be appended to the
      figprefix given.
    """

    xbins = 51

    fig = plt.figure(figsize=(24,18)) # (30,16)
    ax = plt.gca()
    plt.rc('xtick', labelsize=16)    # fontsize of the tick labels
    ax.plot([Ytest.min(), Ytest.max()], [Ytest.min(), Ytest.max()], 'r--', lw=4.)
    plt.hist2d(Ytest, Ypred, bins=xbins, norm=LogNorm())
    cb = plt.colorbar()
    ax.set_xlabel('Observed ' + pred_name, fontsize=38, labelpad=15.)
    ax.set_ylabel('Mean ' + pred_name + ' Predicted', fontsize=38, labelpad=15.)
    ax.axis([Ytest.min()*0.98, Ytest.max()*1.02, Ytest.min()*0.98, Ytest.max()*1.02])
    plt.setp(ax.get_xticklabels(), fontsize=32)
    plt.setp(ax.get_yticklabels(), fontsize=32)
    cb.ax.set_yticklabels(cb.ax.get_yticklabels(), fontsize=28)
    plt.grid(True)
    plt.savefig(figprefix + '_density_predictions.png')
    plt.close()
    print('Generated plot: ', figprefix + '_density_predictions.png')


def plot_2d_density_sigma_vs_error(sigma, yerror, method=None, figprefix=None):
    """Functionality to plot a 2D histogram of the distribution of 
       the standard deviations computed for the predictions vs. the
       computed errors (i.e. values of observed - predicted).
       The plot generated is stored in a png file.

    Parameters
    ----------
    sigma : numpy array
      Array with standard deviations computed.
    yerror : numpy array
      Array with errors computed (observed - predicted).
    method : string
      Method used to comput the standard deviations (i.e. dropout, 
      heteroscedastic, etc.).
    figprefix : string
      String to prefix the filename to store the figure generated.
      A '_density_sigma_error.png' string will be appended to the 
      figprefix given.
    """
    
    xbins = 51
    ybins = 31

    fig = plt.figure(figsize=(24,12)) # (30,16)
    ax = plt.gca()
    plt.rc('xtick', labelsize=16)    # fontsize of the tick labels
    plt.hist2d(sigma, yerror, bins=[xbins,ybins], norm=LogNorm())
    cb = plt.colorbar()
    ax.set_xlabel('Sigma (' + method + ')', fontsize=38, labelpad=15.)
    ax.set_ylabel('Observed - Mean Predicted', fontsize=38, labelpad=15.)
    ax.axis([sigma.min()*0.98, sigma.max()*1.02, -yerror.max(), yerror.max()])
    plt.setp(ax.get_xticklabels(), fontsize=28)
    plt.setp(ax.get_yticklabels(), fontsize=28)
    cb.ax.set_yticklabels(cb.ax.get_yticklabels(), fontsize=22)
    plt.grid(True)
    plt.savefig(figprefix + '_density_sigma_error.png')
    plt.close()
    print('Generated plot: ', figprefix + '_density_sigma_error.png')


def plot_histogram_error_per_sigma(sigma, yerror, method=None, figprefix=None):
    """Functionality to plot a 1D histogram of the distribution of
       computed errors (i.e. values of observed - predicted) observed 
       for specific values of standard deviations computed. The range of
       standard deviations computed is split in xbins values and the 
       1D histograms of error distributions for the smallest six
       standard deviations are plotted.
       The plot generated is stored in a png file.

    Parameters
    ----------
    sigma : numpy array
      Array with standard deviations computed.
    yerror : numpy array
      Array with errors computed (observed - predicted).
    method : string
      Method used to comput the standard deviations (i.e. dropout, 
      heteroscedastic, etc.).
    figprefix : string
      String to prefix the filename to store the figure generated.
      A '_histogram_error_per_sigma.png' string will be appended to 
      the figprefix given.
    """
    
    xbins = 21
    ybins = 31

    H, xedges, yedges, img = plt.hist2d(sigma, yerror,# normed=True,
                                        bins=[xbins,ybins])

    fig = plt.figure(figsize=(14,16))
    legend = []
    for ii in range(6):#(H.shape[0]):
        if ii is not 1:
            plt.plot(yedges[0:H.shape[1]], H[ii,:]/np.sum(H[ii,:]), marker='o',
                 markersize=12, lw=6.)
        legend.append(str((xedges[ii] + xedges[ii+1])/2))
    plt.legend(legend, fontsize=16)
    ax = plt.gca()
    plt.title('Error Dist. per Sigma for ' + method, fontsize=40)
    ax.set_xlabel('Observed - Mean Predicted', fontsize=38, labelpad=15.)
    ax.set_ylabel('Density', fontsize=38, labelpad=15.)
    plt.setp(ax.get_xticklabels(), fontsize=28)
    plt.setp(ax.get_yticklabels(), fontsize=28)
    plt.grid(True)
    plt.savefig(figprefix + '_histogram_error_per_sigma.png')
    plt.close()
    print('Generated plot: ', figprefix + '_histogram_error_per_sigma.png')


def plot_calibration_and_errors(mean_sigma, sigma_start_index, sigma_end_index,
                                min_sigma, max_sigma,
                                error_thresholds,
                                error_thresholds_smooth,
                                err_err,
                                s_interpolate,
                                coverage_percentile,
                                method=None, figprefix=None,
                                steps=False):
    """Functionality to plot empirical calibration curves
       estimated by binning the statistics of computed
       standard deviations and errors.

    Parameters
    ----------
    mean_sigma : numpy array
      Array with the mean standard deviations computed per bin.
    sigma_start_index : non-negative integer
      Index of the mean_sigma array that defines the start of
      the valid empirical calibration interval (i.e. index to
      the smallest std for which a meaningful error is obtained).
    sigma_end_index : non-negative integer
      Index of the mean_sigma array that defines the end of
      the valid empirical calibration interval (i.e. index to
      the largest std for which a meaningful error is obtained).
    min_sigma : numpy array
      Array with the minimum standard deviations computed per bin.
    max_sigma : numpy array
      Array with the maximum standard deviations computed per bin.
    error_thresholds : numpy array
      Thresholds of the errors computed to attain a certain
      error coverage per bin.
    error_thresholds_smooth : numpy array
      Thresholds of the errors computed to attain a certain
      error coverage per bin after a smoothed operation is applied
      to the frequently noisy bin-based estimations.
    err_err : numpy array
      Vertical error bars (usually one standard deviation for a binomial
      distribution estimated by bin) for the error calibration
      computed empirically.
    s_interpolate : scipy.interpolate python object
      A python object from scipy.interpolate that computes a
      univariate spline (InterpolatedUnivariateSpline) constructed
      to express the mapping from standard deviation to error. This 
      spline is generated during the computational empirical 
      calibration procedure.
    coverage_percentile : float
      Value used for the coverage in the percentile estimation
      of the observed error.
    method : string
      Method used to comput the standard deviations (i.e. dropout, 
      heteroscedastic, etc.).
    figprefix : string
      String to prefix the filename to store the figure generated.
      A '_empirical_calibration.png' string will be appended to
      the figprefix given.
    steps : boolean
      Besides the complete empirical calibration (including raw 
      statistics, error bars and smoothing), also generates partial 
      plots with only the raw bin statistics (step1) and with only
      the raw bin statistics and the smoothing interpolation (step2).
    """
    
    xp23 = np.linspace(mean_sigma[sigma_start_index], mean_sigma[sigma_end_index], 200)
    yp23 = s_interpolate(xp23)
        
    p_cov = coverage_percentile
    if steps:
        # Plot raw bin statistics
        fig = plt.figure(figsize=(18,12))
        ax = plt.gca()
        ax.errorbar(mean_sigma, error_thresholds,
            yerr=err_err,
            xerr=[mean_sigma-min_sigma, max_sigma-mean_sigma],
            fmt='o', ecolor='k', capthick=2, ms=8)
        plt.xlabel('Sigma Predicted (' + method + ')', fontsize=24.)
        plt.ylabel(str(p_cov) + '% Coverage for ABS Observed - Mean Predicted', fontsize=24.)
        plt.title('Calibration', fontsize=28)
        ax.axis([0, np.max(max_sigma)*1.1, np.min(error_thresholds)*0.9, np.max(yp23)*1.2])
        plt.grid()
        plt.setp(ax.get_xticklabels(), fontsize=22)
        plt.setp(ax.get_yticklabels(), fontsize=22)
        plt.savefig(figprefix + '_empirical_calibration_step1.png')
        plt.close()
        print('Generated plot: ', figprefix + '_empirical_calibration_step1.png')
        # Plot raw bin statistics and smoothing
        fig = plt.figure(figsize=(18,12))
        ax = plt.gca()
        ax.plot(mean_sigma, error_thresholds_smooth, 'g^', ms=12)
        ax.errorbar(mean_sigma, error_thresholds,
            yerr=err_err,
            xerr=[mean_sigma-min_sigma, max_sigma-mean_sigma],
            fmt='o', ecolor='k', capthick=2, ms=8)
        plt.xlabel('Sigma Predicted (' + method + ')', fontsize=24.)
        plt.ylabel(str(p_cov) + '% Coverage for ABS Observed - Mean Predicted', fontsize=24.)
        plt.title('Calibration', fontsize=28)
        ax.axis([0, np.max(max_sigma)*1.1, np.min(error_thresholds)*0.9, np.max(yp23)*1.2])
        plt.grid()
        plt.setp(ax.get_xticklabels(), fontsize=22)
        plt.setp(ax.get_yticklabels(), fontsize=22)
        plt.savefig(figprefix + '_empirical_calibration_step2.png')
        plt.close()
        print('Generated plot: ', figprefix + '_empirical_calibration_step2.png')

    # Plot raw bin statistics, smoothing and empirical calibration
    fig = plt.figure(figsize=(18,12))
    ax = plt.gca()
    ax.plot(xp23, yp23, 'rx', ms=20)
    ax.plot(mean_sigma, error_thresholds_smooth, 'g^', ms=12)
    ax.errorbar(mean_sigma, error_thresholds,
        yerr=err_err,
        xerr=[mean_sigma-min_sigma, max_sigma-mean_sigma],
        fmt='o', ecolor='k', capthick=2, ms=8)
    plt.xlabel('Sigma Predicted (' + method + ')', fontsize=24.)
    plt.ylabel(str(p_cov) + '% Coverage for ABS Observed - Mean Predicted', fontsize=24.)
    plt.title('Calibration', fontsize=28)
    ax.axis([0, np.max(max_sigma)*1.1, np.min(error_thresholds)*0.9, np.max(yp23)*1.2])
    plt.grid()
    plt.setp(ax.get_xticklabels(), fontsize=22)
    plt.setp(ax.get_yticklabels(), fontsize=22)
    plt.savefig(figprefix + '_empirical_calibration.png')
    plt.close()
    print('Generated plot: ', figprefix + '_empirical_calibration.png')


def plot_percentile_predictions(Ypred, Ypred_Lp, Ypred_Hp, percentile_list, pred_name=None, figprefix=None):
    """Functionality to plot the mean of the percentiles predicted.
       The plot generated is stored in a png file.

    Parameters
    ----------
    Ypred : numpy array
      Array with mid percentile predicted values.
    Ypred_Lp : numpy array
      Array with low percentile predicted values.
    Ypred_Hp : numpy array
      Array with high percentile predicted values.
    percentile_list : string list
      List of percentiles predicted (e.g. '10p', '90p', etc.)
    pred_name : string
      Name of data colum or quantity predicted (e.g. growth, AUC, etc.)
    figprefix : string
      String to prefix the filename to store the figure generated.
      A '_density_predictions.png' string will be appended to the
      figprefix given.
    """

    index_ = np.argsort(Ypred)
    fig = plt.figure(figsize=(24,18))
    plt.scatter(range(index_.shape[0]), Ypred[index_])
    plt.scatter(range(index_.shape[0]), Ypred_Lp[index_])
    plt.scatter(range(index_.shape[0]), Ypred_Hp[index_])
    plt.legend(percentile_list, fontsize=20)
    plt.xlabel('Index', fontsize=18.)
    plt.ylabel(pred_name, fontsize=18.)
    plt.title('Predicted ' + pred_name + ' Percentiles', fontsize=28)
    plt.grid()
    ax = plt.gca()
    plt.setp(ax.get_xticklabels(), fontsize=16)
    plt.setp(ax.get_yticklabels(), fontsize=16)
    plt.savefig(figprefix + '_percentile_predictions.png')
    plt.close()
    print('Generated plot: ', figprefix + '_percentile_predictions.png')


# plot training and validation metrics together and generate one chart per metrics
def plot_metrics(history, title=None, skip_ep=0, outdir='.', add_lr=False):
    """ Plots keras training curves history.
    Args:
        skip_ep: number of epochs to skip when plotting metrics
        add_lr: add curve of learning rate progression over epochs
    """

    def capitalize_metric(met):
        return ' '.join(s.capitalize() for s in met.split('_'))

    all_metrics = list(history.history.keys())
    pr_metrics = ['_'.join(m.split('_')[1:]) for m in all_metrics if 'val' in m]

    epochs = np.asarray(history.epoch) + 1
    if len(epochs) <= skip_ep:
        skip_ep = 0
    eps = epochs[skip_ep:]
    hh = history.history

    for p, m in enumerate(pr_metrics):
        metric_name = m
        metric_name_val = 'val_' + m

        y_tr = hh[metric_name][skip_ep:]
        y_vl = hh[metric_name_val][skip_ep:]

        ymin = min(set(y_tr).union(y_vl))
        ymax = max(set(y_tr).union(y_vl))
        lim = (ymax - ymin) * 0.1
        ymin, ymax = ymin - lim, ymax + lim

        # Start figure
        fig, ax1 = plt.subplots()

        # Plot metrics
        ax1.plot(eps, y_tr, color='b', marker='.', linestyle='-', linewidth=1, alpha=0.6, label=capitalize_metric(metric_name))
        ax1.plot(eps, y_vl, color='r', marker='.', linestyle='--', linewidth=1, alpha=0.6, label=capitalize_metric(metric_name_val))
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel(capitalize_metric(metric_name))
        ax1.set_xlim([min(eps) - 1, max(eps) + 1])
        ax1.set_ylim([ymin, ymax])
        ax1.tick_params('y', colors='k')

        # Add learning rate
        if (add_lr is True) and ('lr' in hh):
            ax2 = ax1.twinx()
            ax2.plot(eps, hh['lr'][skip_ep:], color='g', marker='.', linestyle=':', linewidth=1,
                     alpha=0.6, markersize=5, label='LR')
            ax2.set_ylabel('Learning rate', color='g', fontsize=12)

            ax2.set_yscale('log')
            ax2.tick_params('y', colors='g')

        ax1.grid(True)
        legend = ax1.legend(loc='best', prop={'size': 10})
        frame = legend.get_frame()
        frame.set_facecolor('0.95')
        if title is not None:
            plt.title(title)

        figpath = Path(outdir) / (metric_name + '.png')
        plt.savefig(figpath, bbox_inches='tight')
        plt.close()
