""" 
This code generates CV splits train/val/test splits of a dataset.
"""
from __future__ import print_function, division

import warnings
warnings.filterwarnings('ignore')

import os
import sys
from pathlib import Path
import argparse
from pprint import pprint, pformat
from glob import glob

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

# DATAPATH = '../data/top_21.res_reg.cf_rnaseq.dd_dragon7.labeled.hdf5'
# DATAPATH = '../data/top_21.res_reg.cf_rnaseq.dd_dragon7.labled.parquet'
DATAPATH = '/ccs/home/brettin/Benchmarks/top_21.res_reg.cf_rnaseq.dd_dragon7.labled.hdf5'
outdir = Path('./')
out_splits = outdir/'out_splits'
out_figs = outdir/'out_figs'

# File path
filepath = Path(__file__).resolve().parent

# Utils
# from classlogger import Logger
print_fn = print
# from cv_splitter import cv_splitter, plot_ytr_yvl_dist


def parse_args(args):
    parser = argparse.ArgumentParser(description='Generate and save dataset splits.')

    # Input data
    parser.add_argument('--datapath', default=DATAPATH, type=str, help='Full path to the data.')

    # Feature types
    parser.add_argument('-cf', '--cell_fea', nargs='+', default=['GE'], choices=['GE'], help='Cell features (default: GE).')
    parser.add_argument('-df', '--drug_fea', nargs='+', default=['DD'], choices=['DD'], help='Drug features (default: DD).')    

    # Data split methods
    parser.add_argument('--te_size', type=float, default=0.1, help='Test size split ratio (default: 0.1).')
    # parser.add_argument('--vl_size', type=float, default=0.1, help='Val size split ratio for single split (default: 0.1).')

    parser.add_argument('--split_on', type=str, default=None, choices=['cell', 'drug'], help='Specify how to make a hard split. (default: None).')
    parser.add_argument('--n_splits', type=int, default=100, help='Number of splits to generate (default: 100).')

    # Other
    # parser.add_argument('--seed', type=int, default=0, help='Seed number (Default: 0)')
    parser.add_argument('--n_jobs', default=8,  type=int, help='Default: 8.')

    # Parse args and run
    args = parser.parse_args(args)
    return args


def verify_dirpath(dirpath):
    """ Verify the dirpath exists and contain the dataset. """
    if dirpath is None:
        sys.exit('Program terminated. You must specify a path to a data via the input argument -dp.')

    dirpath = Path(dirpath)
    assert dirpath.exists(), 'The specified dirpath {dirpath} (via argument -dp) was not found.'
    return dirpath


def create_outdir(dirpath, args):
    if args['split_on'] is None:
        outdir = dirpath/'data_splits_seed{}'.format(args['seed'])
    else:
        outdir = dirpath/'data_splits_{}_seed{}'.format(args['split_on'], args['seed'])
    os.makedirs(outdir, exist_ok=True)
    return outdir


def split_size(x):
    """ Split size can be float (0, 1) or int (casts value as needed). """
    assert x > 0, 'Split size must be greater than 0.'
    return int(x) if x > 1.0 else x


def dump_dict(dct, outpath='./dict.txt'):
    """ Dump dict into file. """
    with open( Path(outpath), 'w' ) as file:
        for k in sorted(dct.keys()):
            file.write('{}: {}\n'.format(k, dct[k]))


def get_print_fn(logger):
    """ Returns the python 'print' function if logger is None. Othersiwe, returns logger.info. """
    return print if logger is None else logger.info
    
    
def cnt_fea(df, fea_sep='_', verbose=True, logger=None):
    """ Count the number of features per feature type. """
    print_fn = get_print_fn(logger)

    dct = {}
    unq_prfx = df.columns.map(lambda x: x.split(fea_sep)[0]).unique() # unique feature prefixes
    for prfx in unq_prfx:
        fea_type_cols = [c for c in df.columns if (c.split(fea_sep)[0]) in prfx] # all fea names of specific type
        dct[prfx] = len(fea_type_cols)
    
    if verbose and logger is not None:
        print_fn( pformat(dct) )
    elif verbose:
        pprint(dct)
    return dct


def extract_subset_fea(df, fea_list, fea_sep='_'):
    """ Extract features based feature prefix name. """
    fea = [c for c in df.columns if (c.split(fea_sep)[0]) in fea_list]
    return df[fea]
    
            
def print_intersection_on_var(df, tr_id, vl_id, te_id, grp_col='CELL', logger=None):
    """ Print intersection between train, val, and test datasets with respect
    to grp_col column if provided. df is usually metadata.
    """
    if grp_col in df.columns:
        tr_grp_unq = set(df.loc[tr_id, grp_col])
        vl_grp_unq = set(df.loc[vl_id, grp_col])
        te_grp_unq = set(df.loc[te_id, grp_col])
        print_fn = get_print_fn(logger)

        print_fn(f'\tTotal intersections on {grp_col} btw tr and vl: {len(tr_grp_unq.intersection(vl_grp_unq))}')
        print_fn(f'\tTotal intersections on {grp_col} btw tr and te: {len(tr_grp_unq.intersection(te_grp_unq))}')
        print_fn(f'\tTotal intersections on {grp_col} btw vl and te: {len(vl_grp_unq.intersection(te_grp_unq))}')
        print_fn(f'\tUnique {grp_col} in tr: {len(tr_grp_unq)}')
        print_fn(f'\tUnique {grp_col} in vl: {len(vl_grp_unq)}')
        print_fn(f'\tUnique {grp_col} in te: {len(te_grp_unq)}')    


def plot_hist(x, var_name, fit=None, bins=100, path='hist.png'):
    """ Plot hist of a 1-D array x. """
    if fit is not None:
        (mu, sigma) = stats.norm.fit(x)
        fit = stats.norm
        label = f'norm fit: $\mu$={mu:.2f}, $\sigma$={sigma:.2f}'
    else:
        label = None
    
    alpha = 0.6
    fig, ax = plt.subplots()
    # sns.distplot(x, bins=bins, kde=True, fit=fit, 
    #              hist_kws={'linewidth': 2, 'alpha': alpha, 'color': 'b'},
    #              kde_kws={'linewidth': 2, 'alpha': alpha, 'color': 'k'},
    #              # fit_kws={'linewidth': 2, 'alpha': alpha, 'color': 'r',
    #                       'label': label})
    sns.distplot(x, bins=bins, kde=False, fit=fit, 
                 hist_kws={'linewidth': 2, 'alpha': alpha, 'color': 'b'})
    plt.grid(True)
    if label is not None: plt.legend()
    plt.title(var_name + ' hist')
    plt.savefig(path, bbox_inches='tight')


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

        # elif cv_method=='group':
            # pass
            
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


def run(args):
    te_size = split_size(args['te_size'])
    fea_list = args['cell_fea'] + args['drug_fea']

    # Hard split
    split_on = None if args['split_on'] is None else args['split_on'].upper()
    cv_method = 'simple' if split_on is None else 'group'
    te_method = cv_method 

    # TODO: this needs to be improved
    mltype = 'reg'  # required for the splits (stratify in case of classification)
    
    
    # -----------------------------------------------
    #       Create outdir
    # -----------------------------------------------
    # dump_dict(args, outpath=outdir/'data_split_args.txt') # dump args.
    os.makedirs(out_splits, exist_ok=True)
    os.makedirs(out_figs, exist_ok=True)

    
    # -----------------------------------------------
    #       Load and break data
    # -----------------------------------------------
    print_fn('\nLoad master dataset.')
    if args['datapath'].split('.')[-1]=='parquet':
        data = pd.read_parquet( args['datapath'] ) 
    if args['datapath'].split('.')[-1]=='hdf5':
        data = pd.read_hdf5( args['datapath'] ) 
    print_fn('data.shape {}'.format(data.shape))

    print_fn('Total DD: {}'.format( len([c for c in data.columns if 'DD_' in c]) ))
    print_fn('Total GE: {}'.format( len([c for c in data.columns if 'GE_' in c]) ))
    if 'CELL' in data.columns:
        print_fn('Unique cells: {}'.format( data['CELL'].nunique() ))
    if 'DRUG' in data.columns:
        print_fn('Unique drugs: {}'.format( data['DRUG'].nunique() ))
    # cnt_fea(df, fea_sep='_', verbose=True, logger=lg.logger)
    
    if 'AUC' in data.columns:
        plot_hist(data['AUC'], var_name='AUC', fit=None, bins=100, path=out_figs/'AUC_hist_all.png')
    
    
    # -----------------------------------------------
    #       Generate Hold-Out split (train/val/test)
    # -----------------------------------------------
    """ First, we split the data into train and test. The remaining of train set is further
    splitted into train and validation.
    """
    print_fn('\n{}'.format('-'*50))
    print_fn('Split into hold-out train/val/test')
    print_fn('{}'.format('-'*50))

    n_splits = args['n_splits']
    for seed in range( n_splits ):
        # gen_data_splits.main([ '--seed', str(seed), *args ]) 
        digits = len(str(n_splits))
        seed_str = f"{seed}".zfill(digits)
        output = '1fold_s' + seed_str 

        # Note that we don't shuffle the original dataset, but rather we create a vector array of
        # representative indices.
        np.random.seed( seed )
        idx_vec = np.random.permutation(data.shape[0])
        
        # Create splitter that splits the full dataset into tr and te
        te_folds = int(1/te_size)
        te_splitter = cv_splitter(cv_method=te_method, cv_folds=te_folds, test_size=None,
                                  mltype=mltype, shuffle=False, random_state=seed)
        
        te_grp = None if split_on is None else data[split_on].values[idx_vec]
        if is_string_dtype(te_grp): te_grp = LabelEncoder().fit_transform(te_grp)
        
        # Split tr into tr and te
        tr_id, te_id = next(te_splitter.split(idx_vec, groups=te_grp))
        tr_id = idx_vec[tr_id] # adjust the indices! we'll split the remaining tr into te and vl
        te_id = idx_vec[te_id] # adjust the indices!

        # Update a vector array that excludes the test indices
        idx_vec_ = tr_id; del tr_id

        # Define vl_size while considering the new full size of the available samples
        vl_size = te_size / (1 - te_size)
        cv_folds = int(1/vl_size)

        # Create splitter that splits tr into tr and vl
        cv = cv_splitter(cv_method=cv_method, cv_folds=cv_folds, test_size=None,
                         mltype=mltype, shuffle=False, random_state=seed)    
        
        cv_grp = None if split_on is None else data[split_on].values[idx_vec_]
        if is_string_dtype(cv_grp): cv_grp = LabelEncoder().fit_transform(cv_grp)
        
        # Split tr into tr and vl
        tr_id, vl_id = next(cv.split(idx_vec_, groups=cv_grp))
        tr_id = idx_vec_[tr_id] # adjust the indices!
        vl_id = idx_vec_[vl_id] # adjust the indices!
        
        # Dump tr, vl, te indices
        np.savetxt(out_splits/f'{output}_tr_id.csv', tr_id.reshape(-1,1), fmt='%d', delimiter='', newline='\n')
        np.savetxt(out_splits/f'{output}_vl_id.csv', vl_id.reshape(-1,1), fmt='%d', delimiter='', newline='\n')
        np.savetxt(out_splits/f'{output}_te_id.csv', te_id.reshape(-1,1), fmt='%d', delimiter='', newline='\n')
        
        # Check that indices do not overlap
        assert len( set(tr_id).intersection(set(vl_id)) ) == 0, 'Overlapping indices btw tr and vl'
        assert len( set(tr_id).intersection(set(te_id)) ) == 0, 'Overlapping indices btw tr and te'
        assert len( set(vl_id).intersection(set(te_id)) ) == 0, 'Overlapping indices btw tr and vl'
        
        print_fn('Train samples {} ({:.2f}%)'.format( len(tr_id), 100*len(tr_id)/data.shape[0] ))
        print_fn('Val   samples {} ({:.2f}%)'.format( len(vl_id), 100*len(vl_id)/data.shape[0] ))
        print_fn('Test  samples {} ({:.2f}%)'.format( len(te_id), 100*len(te_id)/data.shape[0] ))
        
        # Confirm that group splits are correct (no intersection)
        grp_col = 'CELL' if split_on is None else split_on
        print_intersection_on_var(data, tr_id=tr_id, vl_id=vl_id, te_id=te_id, grp_col=grp_col)

        if 'AUC' in data.columns:
            plot_hist(data.loc[tr_id, 'AUC'], var_name='AUC', fit=None, bins=100, path=out_figs/f'{output}_AUC_hist_train.png')
            plot_hist(data.loc[vl_id, 'AUC'], var_name='AUC', fit=None, bins=100, path=out_figs/f'{output}_AUC_hist_val.png')
            plot_hist(data.loc[te_id, 'AUC'], var_name='AUC', fit=None, bins=100, path=out_figs/f'{output}_AUC_hist_test.png')
                
            plot_ytr_yvl_dist(ytr=data.loc[tr_id, 'AUC'], yvl=data.loc[vl_id, 'AUC'],
                              title='ytr_yvl_dist', outpath=out_figs/f'{output}_ytr_yvl_dist.png')    
    
            
    # lg.kill_logger()
    print('Done.')
    
    
def main(args):
    args = parse_args(args)
    args = vars(args)
    ret = run(args)
    
    
if __name__ == '__main__':
    main(sys.argv[1:])

