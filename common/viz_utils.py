from pathlib import Path
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import numpy as np

from scipy import interpolate


def plot_history(out, history, metric='loss', val=True, title=None, width=8, height=6):
    title = title or 'model {}'.format(metric)
    val_metric = 'val_{}'.format(metric)
    plt.figure(figsize=(width, height))
    plt.plot(history.history[metric], marker='o')
    if val:
        plt.plot(history.history[val_metric], marker='d')
    plt.title(title)
    plt.ylabel(metric)
    plt.xlabel('epoch')
    if val:
        plt.legend(['train_{}'.format(metric), 'val_{}'.format(metric)], loc='upper center')
    else:
        plt.legend(['train_{}'.format(metric)], loc='upper center')
    png = '{}.plot.{}.png'.format(out, metric)
    plt.savefig(png, bbox_inches='tight')
    plt.close()


def plot_scatter(data, classes, out, width=10, height=8):
    cmap = plt.cm.get_cmap('gist_rainbow')
    plt.figure(figsize=(width, height))
    plt.scatter(data[:, 0], data[:, 1], c=classes, cmap=cmap, lw=0.5, edgecolor='black', alpha=0.7)
    plt.colorbar()
    png = '{}.png'.format(out)
    plt.savefig(png, bbox_inches='tight')
    plt.close()


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


def plot_array(nparray, xlabel, ylabel, title, fname):

    plt.figure()
    plt.plot(nparray, lw=3.)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(fname, bbox_inches='tight')
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

    fig = plt.figure(figsize=(24, 18)) # (30, 16)
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
    plt.savefig(figprefix + '_density_predictions.png', bbox_inches='tight')
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

    fig = plt.figure(figsize=(24, 18)) # (30, 16)
    ax = plt.gca()
    plt.rc('xtick', labelsize=16)    # fontsize of the tick labels
    plt.hist2d(sigma, yerror, bins=[xbins, ybins], norm=LogNorm())
    cb = plt.colorbar()
    ax.set_xlabel('Standard Deviation (' + method + ')', fontsize=38, labelpad=15.)
    ax.set_ylabel('Error: Observed - Mean Predicted', fontsize=38, labelpad=15.)
    ax.axis([sigma.min()*0.98, sigma.max()*1.02, -yerror.max(), yerror.max()])
    plt.setp(ax.get_xticklabels(), fontsize=32)
    plt.setp(ax.get_yticklabels(), fontsize=32)
    cb.ax.set_yticklabels(cb.ax.get_yticklabels(), fontsize=28)
    plt.grid(True)
    plt.savefig(figprefix + '_density_std_error.png', bbox_inches='tight')
    plt.close()
    print('Generated plot: ', figprefix + '_density_std_error.png')


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
                                        bins=[xbins, ybins])

    fig = plt.figure(figsize=(18, 24))
    legend = []
    for ii in range(4):#(H.shape[0]):
        if ii != 1:
            plt.plot(yedges[0:H.shape[1]], H[ii, :]/np.sum(H[ii, :]),
                     marker='o', markersize=12, lw=6.)
        legend.append(str((xedges[ii] + xedges[ii+1])/2))
    plt.legend(legend, fontsize=28)
    ax = plt.gca()
    plt.title('Error Dist. per Standard Deviation for ' + method, fontsize=40)
    ax.set_xlabel('Error: Observed - Mean Predicted', fontsize=38, labelpad=15.)
    ax.set_ylabel('Density', fontsize=38, labelpad=15.)
    plt.setp(ax.get_xticklabels(), fontsize=32)
    plt.setp(ax.get_yticklabels(), fontsize=32)
    plt.grid(True)
    plt.savefig(figprefix + '_histogram_error_per_std.png', bbox_inches='tight')
    plt.close()
    print('Generated plot: ', figprefix + '_histogram_error_per_std.png')


def plot_decile_predictions(Ypred, Ypred_Lp, Ypred_Hp, decile_list, pred_name=None, figprefix=None):
    """Functionality to plot the mean of the deciles predicted.
       The plot generated is stored in a png file.

    Parameters
    ----------
    Ypred : numpy array
      Array with median predicted values.
    Ypred_Lp : numpy array
      Array with low decile predicted values.
    Ypred_Hp : numpy array
      Array with high decile predicted values.
    decile_list : string list
      List of deciles predicted (e.g. '1st', '9th', etc.)
    pred_name : string
      Name of data colum or quantity predicted (e.g. growth, AUC, etc.)
    figprefix : string
      String to prefix the filename to store the figure generated.
      A '_decile_predictions.png' string will be appended to the
      figprefix given.
    """

    index_ = np.argsort(Ypred)
    fig = plt.figure(figsize=(24, 18))
    plt.scatter(range(index_.shape[0]), Ypred[index_])
    plt.scatter(range(index_.shape[0]), Ypred_Lp[index_])
    plt.scatter(range(index_.shape[0]), Ypred_Hp[index_])
    plt.legend(decile_list, fontsize=28)
    plt.xlabel('Index', fontsize=38.)
    plt.ylabel(pred_name, fontsize=38.)
    plt.title('Predicted ' + pred_name + ' Deciles', fontsize=40)
    plt.grid()
    ax = plt.gca()
    plt.setp(ax.get_xticklabels(), fontsize=32)
    plt.setp(ax.get_yticklabels(), fontsize=32)
    plt.savefig(figprefix + '_decile_predictions.png', bbox_inches='tight')
    plt.close()
    print('Generated plot: ', figprefix + '_decile_predictions.png')


def plot_calibration_interpolation(mean_sigma, error, splineobj1, splineobj2, method='', figprefix=None, steps=False):
    """Functionality to plot empirical calibration curves
       estimated by interpolation of the computed
       standard deviations and errors. Since the estimations
       are very noisy, two levels of smoothing are used. Both
       can be plotted independently, if requested.
       The plot(s) generated is(are) stored in png file(s).

    Parameters
    ----------
    mean_sigma : numpy array
      Array with the mean standard deviations computed in inference.
    error : numpy array
      Array with the errors computed from the means predicted in inference.
    splineobj1 : scipy.interpolate python object
      A python object from scipy.interpolate that computes a
      cubic Hermite spline (PchipInterpolator) to express
      the interpolation after the first smoothing. This
      spline is a partial result generated during the empirical
      calibration procedure.
    splineobj2 : scipy.interpolate python object
      A python object from scipy.interpolate that computes a
      cubic Hermite spline (PchipInterpolator) to express
      the mapping from standard deviation to error. This
      spline is generated for interpolating the predictions
      after a process of smoothing-interpolation-smoothing
      computed during the empirical calibration procedure.
    method : string
      Method used to comput the standard deviations (i.e. dropout,
      heteroscedastic, etc.).
    figprefix : string
      String to prefix the filename to store the figure generated.
      A '_empirical_calibration_interpolation.png' string will be appended to
      the figprefix given.
    steps : boolean
      Besides the complete empirical calibration (including the interpolating
      spline), also generates partial plots with only the spline of
      the interpolating spline after the first smoothing level (smooth1).
    """

    xmax = np.max(mean_sigma)
    xmin = np.min(mean_sigma)
    xp23 = np.linspace(xmin, xmax, 200)
    yp23 = splineobj2(xp23)

    if steps:
        # Plot first smoothing
        yp23_1 = splineobj1(xp23)
        fig = plt.figure(figsize=(24, 18))
        ax = plt.gca()
        ax.plot(mean_sigma, error, 'kx')
        ax.plot(xp23, yp23_1, 'gx', ms=20)
        plt.legend(['True', 'Cubic Spline'], fontsize=28)
        plt.xlabel('Standard Deviation Predicted (' + method + ')', fontsize=38.)
        plt.ylabel('Error: ABS Observed - Mean Predicted', fontsize=38.)
        plt.title('Calibration (by Interpolation)', fontsize=40)
        plt.setp(ax.get_xticklabels(), fontsize=32)
        plt.setp(ax.get_yticklabels(), fontsize=32)
        plt.grid()
        fig.tight_layout()
        plt.savefig(figprefix + '_empirical_calibration_interp_smooth1.png', bbox_inches='tight')
        plt.close()
        print('Generated plot: ', figprefix + '_empirical_calibration_interp_smooth1.png')

    fig = plt.figure(figsize=(24, 18))
    ax = plt.gca()
    ax.plot(mean_sigma, error, 'kx')
    ax.plot(xp23, yp23, 'rx', ms=20)
    plt.legend(['True', 'Cubic Spline'], fontsize=28)
    plt.xlabel('Standard Deviation Predicted (' + method + ')', fontsize=38.)
    plt.ylabel('Error: ABS Observed - Mean Predicted', fontsize=38.)
    plt.title('Calibration (by Interpolation)', fontsize=40)
    plt.setp(ax.get_xticklabels(), fontsize=32)
    plt.setp(ax.get_yticklabels(), fontsize=32)
    plt.grid()
    fig.tight_layout()
    plt.savefig(figprefix + '_empirical_calibration_interpolation.png', bbox_inches='tight')
    plt.close()
    print('Generated plot: ', figprefix + '_empirical_calibration_interpolation.png')


def plot_calibrated_std(y_test, y_pred, std_calibrated, thresC, pred_name=None, figprefix=None):
    """Functionality to plot values in testing set after calibration. An estimation of the lower-confidence samples is made. The plot generated is stored in a png file.

    Parameters
    ----------
    y_test : numpy array
      Array with (true) observed values.
    y_pred : numpy array
      Array with predicted values.
    std_calibrated : numpy array
      Array with standard deviation values after calibration.
    thresC : float
      Threshold to label low confidence predictions (low
      confidence predictions are the ones with std > thresC).
    pred_name : string
      Name of data colum or quantity predicted (e.g. growth, AUC, etc.).
    figprefix : string
      String to prefix the filename to store the figure generated.
      A '_calibrated.png' string will be appended to the
      figprefix given.
    """

    N = y_test.shape[0]
    index = np.argsort(y_pred)
    x = np.array(range(N))

    indexC = std_calibrated > thresC
    alphafill = 0.5
    if N > 2000:
        alphafill = 0.7

    scale = 120
    fig = plt.figure(figsize=(24, 18))
    ax = plt.gca()
    ax.scatter(x, y_test[index], color='red', s=scale, alpha=0.5)
    plt.fill_between(x, y_pred[index] - 1.28 * std_calibrated[index],
                     y_pred[index] + 1.28 * std_calibrated[index],
                     color='gray', alpha=alphafill)
    plt.scatter(x, y_pred[index], color='orange', s=scale)
    plt.scatter(x[indexC], y_test[indexC], color='green', s=scale, alpha=0.5)
    plt.legend(['True', '1.28 Std', 'Pred', 'Low conf'], fontsize=28)
    plt.xlabel('Index', fontsize=38.)
    plt.ylabel(pred_name + ' Predicted', fontsize=38.)
    plt.title('Calibrated Standard Deviation', fontsize=40)
    plt.setp(ax.get_xticklabels(), fontsize=32)
    plt.setp(ax.get_yticklabels(), fontsize=32)
    plt.grid()
    fig.tight_layout()
    plt.savefig(figprefix + '_calibrated.png', bbox_inches='tight')
    plt.close()
    print('Generated plot: ', figprefix + '_calibrated.png')


def plot_contamination(y_true, y_pred, sigma, T=None, thresC=0.1, pred_name=None, figprefix=None):
    """Functionality to plot results for the contamination model.
       This includes the latent variables T if they are given (i.e.
       if the results provided correspond to training results). Global
       parameters for the normal distribution are used for shading 80%
       confidence interval.
       If results for training (i.e. T available), samples determined to
       be outliers (i.e. samples whose probability of membership to the
       heavy tailed distribution (Cauchy) is greater than the threshold
       given) are highlighted.
       The plot(s) generated is(are) stored in a png file.

    Parameters
    ----------
    y_true : numpy array
      Array with observed values.
    y_pred : numpy array
      Array with predicted values.
    sigma : float
      Standard deviation of the normal distribution.
    T : numpy array
      Array with latent variables (i.e. membership to normal and heavy-tailed
      distributions). If in testing T is not available (i.e. None)
    thresC : float
      Threshold to label outliers (outliers are the ones
      with probability of membership to heavy-tailed distribution,
      i.e. T[:,1] > thresC).
    pred_name : string
      Name of data colum or quantity predicted (e.g. growth, AUC, etc.).
    figprefix : string
      String to prefix the filename to store the figures generated.
      A '_contamination.png' string will be appended to the
      figprefix given.
    """

    N = y_true.shape[0]
    index = np.argsort(y_pred)
    x = np.array(range(N))

    if T is not None:
        indexG = T[:, 0] > (1. - thresC)
        indexC = T[:, 1] > thresC
        ss = sigma * indexG
        prefig = '_outTrain'
    else:
        ss = sigma
        prefig = '_outTest'
    auxGh = y_pred + 1.28 * ss
    auxGl = y_pred - 1.28 * ss

    # Plotting Outliers
    scale = 120
    fig = plt.figure(figsize=(24, 18))
    ax = plt.gca()
    ax.scatter(x, y_true[index], color='red', s=scale)
    if T is not None:
        plt.scatter(x[indexC], y_true[indexC], color='green', s=scale)  #, alpha=0.8)
    plt.scatter(x, y_pred[index], color='orange', s=scale)
    plt.fill_between(x, auxGl[index], auxGh[index], color='gray', alpha=0.5)
    if T is not None:
        plt.legend(['True', 'Outlier', 'Pred', '1.28 Std'], fontsize=28)
    else:
        plt.legend(['True', 'Pred', '1.28 Std'], fontsize=28)
    plt.xlabel('Index', fontsize=38.)
    plt.ylabel(pred_name + ' Predicted', fontsize=38.)
    plt.title('Contamination Results', fontsize=40)
    plt.setp(ax.get_xticklabels(), fontsize=32)
    plt.setp(ax.get_yticklabels(), fontsize=32)
    plt.grid()
    fig.tight_layout()
    plt.savefig(figprefix + prefig + '_contamination.png', bbox_inches='tight')
    plt.close()
    print('Generated plot: ', figprefix + prefig + '_contamination.png')

    if T is not None:
        # Plotting Latent Variables vs error
        error = np.abs(y_true - y_pred)
        fig = plt.figure(figsize=(24, 18))
        ax = plt.gca()
        ax.scatter(error, T[:, 0], color='blue', s=scale)
        ax.scatter(error, T[:, 1], color='orange', s=scale)
        plt.legend(['Normal', 'Heavy-Tailed'], fontsize=28)
        plt.xlabel('ABS Error', fontsize=38.)
        plt.ylabel('Membership Probability', fontsize=38.)
        plt.title('Contamination: Latent Variables', fontsize=40)
        plt.setp(ax.get_xticklabels(), fontsize=32)
        plt.setp(ax.get_yticklabels(), fontsize=32)
        plt.grid()
        fig.tight_layout()
        plt.savefig(figprefix + '_T_contamination.png', bbox_inches='tight')
        plt.close()
        print('Generated plot: ', figprefix + '_T_contamination.png')


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
