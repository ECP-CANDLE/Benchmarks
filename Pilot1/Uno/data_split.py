from __future__ import print_function, division

import warnings
warnings.filterwarnings('ignore')

import os
import sys
import platform
import argparse
from pathlib import Path
from pprint import pprint, pformat

import sklearn
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from pandas.api.types import is_string_dtype
from sklearn.preprocessing import LabelEncoder

SEED = 0

# DATAPATH = '../top_21.res_reg.cf_rnaseq.dd_dragon7.labeled.hdf5'
DATAPATH = './top_21.res_reg.cf_rnaseq.dd_dragon7.labled.parquet'
outdir = Path('./')

# File path
filepath = Path(__file__).resolve().parent


# Utils
# from classlogger import Logger
# from cv_splitter import cv_splitter, plot_ytr_yvl_dist




def parse_args(args):
    parser = argparse.ArgumentParser(description="Generate data partitions.")
    """
    args['cell_fea'] = 'GE'
    args['drug_fea'] = 'DD'
    args['te_method'] = 'simple'
    args['cv_method'] = 'simple'
    args['te_size'] = 0.1
    args['vl_size'] = 0.1
    """
    parser.add_argument('--seed', default=0, type=int, help='Seed values (default: 0).')
    parser.add_argument('--output', default='1fold', type=str, help='Output name (default: ).')

    # Input data
    # parser.add_argument('--dirpath', default=None, type=str, help='Full path to data and splits (default: None).')
    
    # Select target to predict
    # parser.add_argument('-t', '--target_name', default='AUC', type=str, choices=['AUC'], help='Name of target variable (default: AUC).')

    # Select feature types
    # parser.add_argument('-cf', '--cell_fea', nargs='+', default=['GE'], choices=['GE'], help='Cell line features (default: rna).')
    # parser.add_argument('-df', '--drug_fea', nargs='+', default=['DD'], choices=['DD'], help='Drug features (default: dsc).')

    # Data split methods
    # parser.add_argument('-cvm', '--cv_method', default='simple', type=str, choices=['simple', 'group'], help='CV split method (default: simple).')
    # parser.add_argument('-cvf', '--cv_folds', default=1, type=str, help='Number cross-val folds (default: 1).')
    
    # ML models
    # parser.add_argument('-frm', '--framework', default='lightgbm', type=str, choices=['keras', 'lightgbm', 'sklearn'], help='ML framework (default: lightgbm).')
    # parser.add_argument('-ml', '--model_name', default='nn_reg0', choices=['lgb_reg', 'nn_reg0'], type=str, help='ML model for training (default: nn_reg0).')

    # NN hyper_params
    # parser.add_argument('-ep', '--epochs', default=200, type=int, help='Number of epochs (default: 200).')
    # parser.add_argument('--batch_size', default=32, type=int, help='Batch size (default: 32).')
    # parser.add_argument('--dr_rate', default=0.2, type=float, help='Dropout rate (default: 0.2).')
    # parser.add_argument('-sc', '--scaler', default='stnd', type=str, choices=['stnd', 'minmax', 'rbst'], help='Feature normalization method (default: stnd).')

    # parser.add_argument('--opt', default='sgd', type=str, choices=['sgd', 'adam'], help='Optimizer name (default: sgd).')

    # parser.add_argument('--clr_mode', default=None, type=str, choices=['trng1', 'trng2', 'exp'], help='CLR mode (default: trng1).')
    # parser.add_argument('--clr_base_lr', type=float, default=1e-4, help='Base lr for cycle lr.')
    # parser.add_argument('--clr_max_lr', type=float, default=1e-3, help='Max lr for cycle lr.')
    # parser.add_argument('--clr_gamma', type=float, default=0.999994, help='Gamma parameter for learning cycle LR.')

    # Define n_jobs
    # parser.add_argument('--skp_ep', default=3, type=int, help='Default: 3.')
    # parser.add_argument('--n_jobs', default=4, type=int, help='Default: 4.')

    # Parse args
    args = parser.parse_args(args)
    return args



# -----------------------------------------------------------
# Code below comes from cv_splitter.py
# -----------------------------------------------------------
from sklearn.model_selection import ShuffleSplit, KFold
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold


def cv_splitter(cv_method: str='simple', cv_folds: int=1, test_size: float=0.2,
                mltype: str='reg', shuffle: bool=False, random_state=None):
    """ Creates a cross-validation splitter.
    Args:
        cv_method: 'simple', 'stratify' (only for classification), 'groups' (only for regression)
        cv_folds: number of cv folds
        test_size: fraction of test set size (used only if cv_folds=1)
        mltype: 'reg', 'cls'
    """
    # Classification
    if mltype == 'cls':
        if cv_method == 'simple':
            if cv_folds == 1:
                cv = ShuffleSplit(n_splits=cv_folds, test_size=test_size, random_state=random_state)
            else:
                cv = KFold(n_splits=cv_folds, shuffle=shuffle, random_state=random_state)
            
        elif cv_method == 'stratify':
            if cv_folds == 1:
                cv = StratifiedShuffleSplit(n_splits=cv_folds, test_size=test_size, random_state=random_state)
            else:
                cv = StratifiedKFold(n_splits=cv_folds, shuffle=shuffle, random_state=random_state)

    # Regression
    elif mltype == 'reg':
        # Regression
        if cv_method == 'group':
            if cv_folds == 1:
                cv = GroupShuffleSplit(n_splits=cv_folds, random_state=random_state)
            else:
                cv = GroupKFold(n_splits=cv_folds)
            
        elif cv_method == 'simple':
            if cv_folds == 1:
                cv = ShuffleSplit(n_splits=cv_folds, test_size=test_size, random_state=random_state)
            else:
                cv = KFold(n_splits=cv_folds, shuffle=shuffle, random_state=random_state)
    return cv


def plot_ytr_yvl_dist(ytr, yvl, title=None, outpath='.'):
    """ Plot distributions of response data of train and val sets. """
    fig, ax = plt.subplots()
    plt.hist(ytr, bins=100, label='ytr', color='b', alpha=0.5)
    plt.hist(yvl, bins=100, label='yvl', color='r', alpha=0.5)
    if title is None: title = ''
    plt.title(title)
    plt.tight_layout()
    plt.grid(True)
    plt.legend()
    if outpath is None:
        plt.savefig(Path(outpath)/'ytr_yvl_dist.png', bbox_inches='tight')
    else:
        plt.savefig(outpath, bbox_inches='tight')
# -----------------------------------------------------------


        
def split_size(x):
    """ Split size can be float (0, 1) or int (casts value as needed). """
    assert x > 0, 'Split size must be greater than 0.'
    return int(x) if x > 1.0 else x


def dump_dict(dct, outpath='./dict.txt'):
    """ Dump dict into file. """
    with open( Path(outpath), 'w' ) as file:
        for k in sorted(dct.keys()):
            file.write('{}: {}\n'.format(k, dct[k]))

            
def plot_hist(x, var_name, fit=None, bins=100, path='hist.png'):
    """ Plot hist of a 1-D array x. """
    if fit is not None:
        (mu, sigma) = stats.norm.fit(x)
        fit = stats.norm
        label = f'norm fit: $\mu$={mu:.2f}, $\sigma$={sigma:.2f}'
    else:
        label = ''
    
    alpha = 0.6
    fig, ax = plt.subplots()
#     sns.distplot(x, bins=bins, kde=True, fit=fit, 
#                  hist_kws={'linewidth': 2, 'alpha': alpha, 'color': 'b'},
#                  kde_kws={'linewidth': 2, 'alpha': alpha, 'color': 'k'},
#                  fit_kws={'linewidth': 2, 'alpha': alpha, 'color': 'r',
#                           'label': label})
    sns.distplot(x, bins=bins, kde=False, fit=fit, 
                 hist_kws={'linewidth': 2, 'alpha': alpha, 'color': 'b'})
    plt.grid(True)
    plt.legend()
    plt.title(var_name + ' hist')
    plt.savefig(path, bbox_inches='tight')
    
            
# def make_split(xdata, meta, outdir, args):
def make_split(args):

    # Load data
    data = pd.read_parquet( DATAPATH ) 

    # Data splits
    # te_method = args['te_method']
    # cv_method = args['cv_method']
    # te_size = split_size(args['te_size'])
    # vl_size = split_size(args['vl_size'])

    seed = args['seed']
    output = args['output']

    te_method = 'simple'
    cv_method = 'simple'
    te_size = 0.1
    vl_size = 0.1

    # Features 
    # cell_fea = args['cell_fea']
    # drug_fea = args['drug_fea']
    cell_fea = 'GE'
    drug_fea = 'DD'
    # fea_list = cell_fea + drug_fea
    
    # Other params
    # n_jobs = args['n_jobs']
    n_jobs = 8

    # Hard split
    grp_by_col = None
    # cv_method = 'simple'

    # TODO: this need to be improved
    mltype = 'reg'  # required for the splits (stratify in case of classification)
    
    
    # -----------------------------------------------
    #       Outdir and Logger
    # -----------------------------------------------
    # Logger
    # lg = Logger(outdir/'splitter.log')
    # lg.logger.info(f'File path: {filepath}')
    # lg.logger.info(f'\n{pformat(args)}')

    # Dump args to file
    # dump_dict(args, outpath=outdir/'args.txt')


    # -----------------------------------------------
    #       Load data and pre-proc
    # -----------------------------------------------
    # if (outdir/'xdata.parquet').is_file():
    #     xdata = pd.read_parquet( outdir/'xdata.parquet' )
    #     meta = pd.read_parquet( outdir/'meta.parquet' )
    
    # lg.logger.info('Totoal DD: {}'.format( len([c for c in xdata.columns if 'DD' in c]) ))
    # lg.logger.info('Totoal GE: {}'.format( len([c for c in xdata.columns if 'GE' in c]) ))
    # lg.logger.info('Unique cells: {}'.format( meta['CELL'].nunique() ))
    # lg.logger.info('Unique drugs: {}'.format( meta['DRUG'].nunique() ))

    # plot_hist(meta['AUC'], var_name='AUC', fit=None, bins=100, path=outdir/'AUC_hist_all.png')
    
    
    # -----------------------------------------------
    #       Train-test split
    # -----------------------------------------------
    # np.random.seed(SEED)
    np.random.seed(seed)
    # idx_vec = np.random.permutation(xdata.shape[0])
    idx_vec = np.random.permutation(data.shape[0])

    if te_method is not None:
        # lg.logger.info('\nSplit train/test.')
        te_splitter = cv_splitter(cv_method=te_method, cv_folds=1, test_size=te_size,
                                  mltype=mltype, shuffle=False, random_state=seed)

        te_grp = meta[grp_by_col].values[idx_vec] if te_method=='group' else None
        if is_string_dtype(te_grp): te_grp = LabelEncoder().fit_transform(te_grp)
   
        # Split train/test
        tr_id, te_id = next(te_splitter.split(idx_vec, groups=te_grp))
        tr_id = idx_vec[tr_id] # adjust the indices!
        te_id = idx_vec[te_id] # adjust the indices!

        pd.Series(tr_id).to_csv(outdir/f'{output}_tr_id.csv', index=False, header=[0])
        # pd.Series(te_id).to_csv(outdir/f'{output}_te_id.csv', index=False, header=[0])
        pd.Series(te_id).to_csv(outdir/f'{output}_vl_id.csv', index=False, header=[0])
        
        # lg.logger.info('Train: {:.1f}'.format( len(tr_id)/xdata.shape[0] ))
        # lg.logger.info('Test:  {:.1f}'.format( len(te_id)/xdata.shape[0] ))
        
        # Update the master idx vector for the CV splits
        # idx_vec = tr_id

        # Plot dist of responses (TODO: this can be done to all response metrics)
        # plot_ytr_yvl_dist(ytr=tr_ydata.values, yvl=te_ydata.values,
        #         title='tr and te', outpath=run_outdir/'tr_te_resp_dist.png')

        # Confirm that group splits are correct
        if te_method=='group' and grp_by_col is not None:
            tr_grp_unq = set(meta.loc[tr_id, grp_by_col])
            te_grp_unq = set(meta.loc[te_id, grp_by_col])
            lg.logger.info(f'\tTotal group ({grp_by_col}) intersections btw tr and te: {len(tr_grp_unq.intersection(te_grp_unq))}.')
            lg.logger.info(f'\tA few intersections : {list(tr_grp_unq.intersection(te_grp_unq))[:3]}.')

        # Update vl_size to effective vl_size
        # vl_size = vl_size * xdata.shape[0]/len(tr_id)
        
        # Plot hist te
        # pd.Series(meta.loc[te_id, 'AUC'].values, name='yte').to_csv(outdir/'yte.csv')
        # plot_hist(meta.loc[te_id, 'AUC'], var_name='AUC', fit=None, bins=100, path=outdir/'AUC_hist_test.png')

        # del tr_id, te_id


    # -----------------------------------------------
    #       Generate CV splits
    # -----------------------------------------------
    """
    # cv_folds_list = [1, 5, 7, 10, 15, 20]
    cv_folds_list = [1]
    lg.logger.info(f'\nStart CV splits ...')
    
    for cv_folds in cv_folds_list:
        lg.logger.info(f'\nCV folds: {cv_folds}')

        cv = cv_splitter(cv_method=cv_method, cv_folds=cv_folds, test_size=vl_size,
                         mltype=mltype, shuffle=False, random_state=SEED)

        cv_grp = meta[grp_by_col].values[idx_vec] if cv_method=='group' else None
        if is_string_dtype(cv_grp): cv_grp = LabelEncoder().fit_transform(cv_grp)
    
        tr_folds = {}
        vl_folds = {}

        # Start CV iters
        for fold, (tr_id, vl_id) in enumerate(cv.split(idx_vec, groups=cv_grp)):
            tr_id = idx_vec[tr_id] # adjust the indices!
            vl_id = idx_vec[vl_id] # adjust the indices!

            tr_folds[fold] = tr_id.tolist()
            vl_folds[fold] = vl_id.tolist()

            # Confirm that group splits are correct
            if cv_method=='group' and grp_by_col is not None:
                tr_grp_unq = set(meta.loc[tr_id, grp_by_col])
                vl_grp_unq = set(meta.loc[vl_id, grp_by_col])
                lg.logger.info(f'\tTotal group ({grp_by_col}) intersections btw tr and vl: {len(tr_grp_unq.intersection(vl_grp_unq))}.')
                lg.logger.info(f'\tUnique cell lines in tr: {len(tr_grp_unq)}.')
                lg.logger.info(f'\tUnique cell lines in vl: {len(vl_grp_unq)}.')
        
        # Convet to df
        # from_dict takes too long  -->  faster described here: stackoverflow.com/questions/19736080/
        # tr_folds = pd.DataFrame.from_dict(tr_folds, orient='index').T 
        # vl_folds = pd.DataFrame.from_dict(vl_folds, orient='index').T
        tr_folds = pd.DataFrame(dict([ (k, pd.Series(v)) for k, v in tr_folds.items() ]))
        vl_folds = pd.DataFrame(dict([ (k, pd.Series(v)) for k, v in vl_folds.items() ]))

        # Dump
        tr_folds.to_csv(outdir/f'{cv_folds}fold_tr_id.csv', index=False)
        vl_folds.to_csv(outdir/f'{cv_folds}fold_vl_id.csv', index=False)
        
        if cv_folds==1 and fold==0:
            plot_hist(meta.loc[tr_id, 'AUC'], var_name='AUC', fit=None, bins=100, path=outdir/'AUC_hist_train.png')
            plot_hist(meta.loc[vl_id, 'AUC'], var_name='AUC', fit=None, bins=100, path=outdir/'AUC_hist_val.png')
            
            plot_ytr_yvl_dist(ytr=meta.loc[tr_id, 'AUC'], yvl=meta.loc[vl_id, 'AUC'],
                              title='ytr_yvl_dist', outpath=outdir/'ytr_yvl_dist.png')
            
            pd.Series(meta.loc[tr_id, 'AUC'].values, name='ytr').to_csv(outdir/'ytr.csv')
            pd.Series(meta.loc[vl_id, 'AUC'].values, name='yvl').to_csv(outdir/'yvl.csv')

    lg.kill_logger()
    # print('Done.')
    """

    

def main(args):
    args = parse_args(args)
    args = vars(args)
    # run(args)
    make_split(args)
    

if __name__ == '__main__':
    main(sys.argv[1:])



