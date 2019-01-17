from __future__ import print_function

import collections
import json
import logging
import os
import pickle
import sys
import mygene

import numpy as np
import pandas as pd

from itertools import cycle, islice

from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler
from sklearn.model_selection import ShuffleSplit, KFold

file_path = os.path.dirname(os.path.realpath(__file__))
lib_path = os.path.abspath(os.path.join(file_path, '..', 'common'))
sys.path.append(lib_path)

import candle_keras as candle

global_cache = {}

SEED = 2018

P1B3_URL = 'http://ftp.mcs.anl.gov/pub/candle/public/benchmarks/P1B3/'
DATA_URL = 'http://ftp.mcs.anl.gov/pub/candle/public/benchmarks/Pilot1/combo/'

logger = logging.getLogger(__name__)


def set_up_logger(verbose=False):
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter(''))
    sh.setLevel(logging.DEBUG if verbose else logging.INFO)

    logger.setLevel(logging.DEBUG)
    logger.addHandler(sh)


def set_seed(seed=SEED):
    os.environ['PYTHONHASHSEED'] = '0'
    np.random.seed(seed)
    random.seed(seed)


def get_file(url):
    fname = os.path.basename(url)
    return candle.get_file(fname, origin=url, cache_subdir='Pilot1')


def impute_and_scale(df, scaling='std', imputing='mean', dropna='all'):
    """Impute missing values with mean and scale data included in pandas dataframe.

    Parameters
    ----------
    df : pandas dataframe
        dataframe to impute and scale
    scaling : 'maxabs' [-1,1], 'minmax' [0,1], 'std', or None, optional (default 'std')
        type of scaling to apply
    """

    if dropna:
        df = df.dropna(axis=1, how=dropna)
    else:
        empty_cols = df.columns[df.notnull().sum() == 0]
        df[empty_cols] = 0

    if imputing is None or imputing.lower() == 'none':
        mat = df.values
    else:
        imputer = Imputer(strategy=imputing, axis=0)
        mat = imputer.fit_transform(df)

    if scaling is None or scaling.lower() == 'none':
        return pd.DataFrame(mat, columns=df.columns)

    if scaling == 'maxabs':
        scaler = MaxAbsScaler()
    elif scaling == 'minmax':
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()

    mat = scaler.fit_transform(mat)
    df = pd.DataFrame(mat, columns=df.columns)

    return df


def discretize(df, col, bins=2, cutoffs=None):
    y = df[col]
    thresholds = cutoffs
    if thresholds is None:
        percentiles = [100 / bins * (i + 1) for i in range(bins - 1)]
        thresholds = [np.percentile(y, x) for x in percentiles]
    classes = np.digitize(y, thresholds)
    df[col] = classes
    return df


def save_combined_dose_response():
    df1 = load_single_dose_response(combo_format=True, fraction=False)
    df2 = load_combo_dose_response(fraction=False)
    df = pd.concat([df1, df2])
    df.to_csv('combined_drug_growth', index=False, sep='\t')


def load_combined_dose_response(rename=True):
    df1 = load_single_dose_response(combo_format=True)
    logger.info('Loaded {} single drug dose response measurements'.format(df1.shape[0]))

    df2 = load_combo_dose_response()
    logger.info('Loaded {} drug pair dose response measurements'.format(df2.shape[0]))

    df = pd.concat([df1, df2])
    logger.info('Combined dose response data contains sources: {}'.format(df['SOURCE'].unique()))

    if rename:
        df = df.rename(columns={'SOURCE': 'Source', 'CELL': 'Sample',
                                'DRUG1': 'Drug1', 'DRUG2': 'Drug2',
                                'DOSE1': 'Dose1', 'DOSE2': 'Dose2',
                                'GROWTH': 'Growth', 'STUDY': 'Study'})
    return df


def load_single_dose_response(combo_format=False, fraction=True):
    # path = get_file(DATA_URL + 'combined_single_drug_growth')
    path = get_file(DATA_URL + 'rescaled_combined_single_drug_growth')

    df = global_cache.get(path)
    if df is None:
        df = pd.read_csv(path, sep='\t', engine='c',
                         na_values=['na', '-', ''],
                         # nrows=10,
                         dtype={'SOURCE': str, 'DRUG_ID': str,
                                'CELLNAME': str, 'CONCUNIT': str,
                                'LOG_CONCENTRATION': np.float32,
                                'EXPID': str, 'GROWTH': np.float32})
        global_cache[path] = df

    df['DOSE'] = -df['LOG_CONCENTRATION']

    df = df.rename(columns={'CELLNAME': 'CELL', 'DRUG_ID': 'DRUG', 'EXPID': 'STUDY'})
    df = df[['SOURCE', 'CELL', 'DRUG', 'DOSE', 'GROWTH', 'STUDY']]

    if fraction:
        df['GROWTH'] /= 100

    if combo_format:
        df = df.rename(columns={'DRUG': 'DRUG1', 'DOSE': 'DOSE1'})
        df['DRUG2'] = np.nan
        df['DOSE2'] = np.nan
        df['DRUG2'] = df['DRUG2'].astype(object)
        df['DOSE2'] = df['DOSE2'].astype(np.float32)
        df = df[['SOURCE', 'CELL', 'DRUG1', 'DOSE1', 'DRUG2', 'DOSE2', 'GROWTH', 'STUDY']]

    return df


def load_combo_dose_response(fraction=True):
    path = get_file(DATA_URL + 'ComboDrugGrowth_Nov2017.csv')
    df = global_cache.get(path)
    if df is None:
        df = pd.read_csv(path, sep=',', engine='c',
                         na_values=['na', '-', ''],
                         usecols=['CELLNAME', 'NSC1', 'CONC1', 'NSC2', 'CONC2',
                                  'PERCENTGROWTH', 'VALID', 'SCREENER', 'STUDY'],
                         # nrows=10000,
                         dtype={'CELLNAME': str, 'NSC1': str, 'NSC2': str,
                                'CONC1': np.float32, 'CONC2': np.float32,
                                'PERCENTGROWTH': np.float32, 'VALID': str,
                                'SCREENER': str, 'STUDY': str},
                         error_bad_lines=False, warn_bad_lines=True)
        global_cache[path] = df

    df = df[df['VALID'] == 'Y']

    df['SOURCE'] = 'ALMANAC.' + df['SCREENER']

    cellmap_path = get_file(DATA_URL + 'NCI60_CELLNAME_to_Combo.txt')
    df_cellmap = pd.read_table(cellmap_path)
    df_cellmap.set_index('Name', inplace=True)
    cellmap = df_cellmap[['NCI60.ID']].to_dict()['NCI60.ID']

    df['CELL'] = df['CELLNAME'].map(lambda x: cellmap[x])

    df['DOSE1'] = -np.log10(df['CONC1'])
    df['DOSE2'] = -np.log10(df['CONC2'])

    df['DRUG1'] = 'NSC.' + df['NSC1']
    df['DRUG2'] = 'NSC.' + df['NSC2']

    if fraction:
        df['GROWTH'] = df['PERCENTGROWTH'] / 100
    else:
        df['GROWTH'] = df['PERCENTGROWTH']

    df = df[['SOURCE', 'CELL', 'DRUG1', 'DOSE1', 'DRUG2', 'DOSE2', 'GROWTH', 'STUDY']]

    return df


def load_drug_data(ncols=None, scaling='std', imputing='mean', dropna=None, add_prefix=True):
    df_info = load_drug_info()
    df_info['Drug'] = df_info['PUBCHEM']

    df_desc = load_drug_set_descriptors(drug_set='Combined_PubChem', ncols=ncols)
    df_fp = load_drug_set_fingerprints(drug_set='Combined_PubChem', ncols=ncols)

    df_desc = pd.merge(df_info[['ID', 'Drug']], df_desc, on='Drug').drop('Drug', 1).rename(columns={'ID': 'Drug'})
    df_fp = pd.merge(df_info[['ID', 'Drug']], df_fp, on='Drug').drop('Drug', 1).rename(columns={'ID': 'Drug'})

    df_desc2 = load_drug_set_descriptors(drug_set='NCI60', usecols=df_desc.columns.tolist() if ncols else None)
    df_fp2 = load_drug_set_fingerprints(drug_set='NCI60', usecols=df_fp.columns.tolist() if ncols else None)

    df_desc = pd.concat([df_desc, df_desc2]).reset_index(drop=True)
    df1 = pd.DataFrame(df_desc.loc[:, 'Drug'])
    df2 = df_desc.drop('Drug', 1)
    df2 = impute_and_scale(df2, scaling=scaling, imputing=imputing, dropna=dropna)
    if add_prefix:
        df2 = df2.add_prefix('dragon7.')
    df_desc = pd.concat([df1, df2], axis=1)

    df_fp = pd.concat([df_fp, df_fp2]).reset_index(drop=True)
    df1 = pd.DataFrame(df_fp.loc[:, 'Drug'])
    df2 = df_fp.drop('Drug', 1)
    df2 = impute_and_scale(df2, scaling=None, imputing=imputing, dropna=dropna)
    if add_prefix:
        df2 = df2.add_prefix('dragon7.')
    df_fp = pd.concat([df1, df2], axis=1)

    logger.info('Loaded combined dragon7 drug descriptors: %s', df_desc.shape)
    logger.info('Loaded combined dragon7 drug fingerprints: %s', df_fp.shape)

    return df_desc, df_fp


def load_drug_descriptors(ncols=None, scaling='std', imputing='mean', dropna=None, add_prefix=True):
    df_info = load_drug_info()
    df_info['Drug'] = df_info['PUBCHEM']

    df_desc = load_drug_set_descriptors(drug_set='Combined_PubChem', ncols=ncols)
    df_desc = pd.merge(df_info[['ID', 'Drug']], df_desc, on='Drug').drop('Drug', 1).rename(columns={'ID': 'Drug'})

    df_desc2 = load_drug_set_descriptors(drug_set='NCI60', usecols=df_desc.columns.tolist() if ncols else None)

    df_desc = pd.concat([df_desc, df_desc2]).reset_index(drop=True)
    df1 = pd.DataFrame(df_desc.loc[:, 'Drug'])
    df2 = df_desc.drop('Drug', 1)
    df2 = impute_and_scale(df2, scaling=scaling, imputing=imputing, dropna=dropna)
    if add_prefix:
        df2 = df2.add_prefix('dragon7.')
    df_desc = pd.concat([df1, df2], axis=1)

    logger.info('Loaded combined dragon7 drug descriptors: %s', df_desc.shape)

    return df_desc


def load_drug_fingerprints(ncols=None, scaling='std', imputing='mean', dropna=None, add_prefix=True):
    df_info = load_drug_info()
    df_info['Drug'] = df_info['PUBCHEM']

    df_fp = load_drug_set_fingerprints(drug_set='Combined_PubChem', ncols=ncols)
    df_fp = pd.merge(df_info[['ID', 'Drug']], df_fp, on='Drug').drop('Drug', 1).rename(columns={'ID': 'Drug'})

    df_fp2 = load_drug_set_fingerprints(drug_set='NCI60', usecols=df_fp.columns.tolist() if ncols else None)

    df_fp = pd.concat([df_fp, df_fp2]).reset_index(drop=True)
    df1 = pd.DataFrame(df_fp.loc[:, 'Drug'])
    df2 = df_fp.drop('Drug', 1)
    df2 = impute_and_scale(df2, scaling=None, imputing=imputing, dropna=dropna)
    if add_prefix:
        df2 = df2.add_prefix('dragon7.')
    df_fp = pd.concat([df1, df2], axis=1)

    logger.info('Loaded combined dragon7 drug fingerprints: %s', df_fp.shape)

    return df_fp


def load_drug_info():
    path = get_file(DATA_URL + 'drug_info')
    df = pd.read_table(path, dtype=object)
    df['PUBCHEM'] = 'PubChem.CID.' + df['PUBCHEM']
    return df


def lookup(df, query, ret, keys, match='match'):
    mask = pd.Series(False, index=range(df.shape[0]))
    for key in keys:
        if match == 'contains':
            mask |= df[key].str.contains(query.upper(), case=False)
        else:
            mask |= (df[key].str.upper() == query.upper())
    return list(set(df[mask][ret].values.flatten().tolist()))


def load_cell_metadata():
    path = get_file(DATA_URL + 'cl_metadata')
    df = pd.read_table(path)
    return df


def cell_name_to_ids(name, source=None):
    path = get_file(DATA_URL + 'NCI60_CELLNAME_to_Combo.txt')
    df1 = pd.read_table(path)
    hits1 = lookup(df1, name, 'NCI60.ID', ['NCI60.ID', 'CELLNAME', 'Name'], match='contains')
    path = get_file(DATA_URL + 'cl_mapping')
    df2 = pd.read_table(path, header=None)
    hits2 = lookup(df2, name, [0, 1], [0, 1], match='contains')
    hits = hits1 + hits2
    if source:
        hits = [x for x in hits if x.startswith(source.upper() + '.')]
    return hits


def drug_name_to_ids(name, source=None):
    df1 = load_drug_info()
    path = get_file(DATA_URL + 'NCI_IOA_AOA_drugs')
    df2 = pd.read_table(path, dtype=str)
    df2['NSC'] = 'NSC.' + df2['NSC']
    hits1 = lookup(df1, name, 'ID', ['ID', 'NAME', 'CLEAN_NAME', 'PUBCHEM'])
    hits2 = lookup(df2, name, 'NSC', ['NSC', 'Generic Name', 'Preffered Name'])
    hits = hits1 + hits2
    if source:
        hits = [x for x in hits if x.startswith(source.upper() + '.')]
    return hits


def load_drug_set_descriptors(drug_set='Combined_PubChem', ncols=None, usecols=None,
                              scaling=None, imputing=None, add_prefix=False):
    path = get_file(DATA_URL + '{}_dragon7_descriptors.tsv'.format(drug_set))

    df_cols = pd.read_table(path, engine='c', nrows=0)
    total = df_cols.shape[1] - 1
    if usecols is not None:
        usecols = [x for x in usecols if x in df_cols.columns]
        if usecols[0] != 'NAME':
            usecols = ['NAME'] + usecols
        df_cols = df_cols.loc[:, usecols]
    elif ncols and ncols < total:
        usecols = np.random.choice(total, size=ncols, replace=False)
        usecols = np.append([0], np.add(sorted(usecols), 1))
        df_cols = df_cols.iloc[:, usecols]

    dtype_dict = dict((x, np.float32) for x in df_cols.columns[1:])
    df = pd.read_table(path, engine='c', usecols=usecols, dtype=dtype_dict,
                       na_values=['na', '-', ''])

    df1 = pd.DataFrame(df.loc[:, 'NAME'])
    df1.rename(columns={'NAME': 'Drug'}, inplace=True)

    df2 = df.drop('NAME', 1)
    if add_prefix:
        df2 = df2.add_prefix('dragon7.')

    df2 = impute_and_scale(df2, scaling, imputing, dropna=None)

    df = pd.concat([df1, df2], axis=1)
    return df


def load_drug_set_fingerprints(drug_set='Combined_PubChem', ncols=None, usecols=None,
                               scaling=None, imputing=None, add_prefix=False):
    fps = ['PFP', 'ECFP']
    usecols_all = usecols
    df_merged = None
    for fp in fps:
        path = get_file(DATA_URL + '{}_dragon7_{}.tsv'.format(drug_set, fp))
        df_cols = pd.read_table(path, engine='c', nrows=0, skiprows=1, header=None)
        total = df_cols.shape[1] - 1
        if usecols_all is not None:
            usecols = [x.replace(fp + '.', '') for x in usecols_all]
            usecols = [int(x) for x in usecols if x.isdigit()]
            usecols = [x for x in usecols if x in df_cols.columns]
            if usecols[0] != 0:
                usecols = [0] + usecols
            df_cols = df_cols.loc[:, usecols]
        elif ncols and ncols < total:
            usecols = np.random.choice(total, size=ncols, replace=False)
            usecols = np.append([0], np.add(sorted(usecols), 1))
            df_cols = df_cols.iloc[:, usecols]

        dtype_dict = dict((x, np.float32) for x in df_cols.columns[1:])
        df = pd.read_table(path, engine='c', skiprows=1, header=None,
                           usecols=usecols, dtype=dtype_dict)
        df.columns = ['{}.{}'.format(fp, x) for x in df.columns]

        col1 = '{}.0'.format(fp)
        df1 = pd.DataFrame(df.loc[:, col1])
        df1.rename(columns={col1: 'Drug'}, inplace=True)

        df2 = df.drop(col1, 1)
        if add_prefix:
            df2 = df2.add_prefix('dragon7.')

        df2 = impute_and_scale(df2, scaling, imputing, dropna=None)

        df = pd.concat([df1, df2], axis=1)

        df_merged = df if df_merged is None else df_merged.merge(df)

    return df_merged


# def load_drug_smiles():
#     path = get_file(DATA_URL + 'ChemStructures_Consistent.smiles')

#     df = global_cache.get(path)
#     if df is None:
#         df = pd.read_csv(path, sep='\t', engine='c', dtype={'nsc_id':object})
#         df = df.rename(columns={'nsc_id': 'NSC'})
#         global_cache[path] = df

#     return df

def encode_sources(sources):
    df = pd.get_dummies(sources, prefix='source', prefix_sep='.')
    df['Source'] = sources
    source_l1 = df['Source'].str.extract('^(\S+)\.', expand=False)
    df1 = pd.get_dummies(source_l1, prefix='source.L1', prefix_sep='.')
    df = pd.concat([df1, df], axis=1)
    df = df.set_index('Source').reset_index()
    return df


def load_genemania_adj_matrix(fp="GraphCovTestPMDR/GeneMania_adj.hdf"):
    df = pd.read_hdf(fp, key="adj")
    return df


def align_features(df, adj, features_to_ensembl="/home/aclyde11/scratch-area/ensembl_name_rnaseq_dict.hdf"):
    print("aligning genemania!")
    df_orig = df.columns.to_series()
    df.columns = df_orig.apply(lambda x: str(x)[7:])
    gene_names = df.columns
    mg = mygene.MyGeneInfo()
    qu = mg.querymany(gene_names, scopes='symbol', fields="ensembl.gene", as_dataframe=True, df_index=True)
    mapper = qu[qu['ensembl.gene'].apply(lambda x: str(x)[:4]) == "ENSG"]['ensembl.gene'].to_dict()
    df = df.rename(mapper, axis=1)
    print("from data")
    print(df.columns)
    print("from gnee")
    print(mapper.keys())
    data_features = set(df.columns)
    adj_features = set(adj.columns)
    common_features = data_features.intersection(adj_features)
    print("There are %i common features" % len(common_features))
    df = df.drop(data_features.difference(common_features), axis=1)
    adj = adj.drop(adj_features.difference(common_features), axis=1).drop(adj_features.difference(common_features),
                                                                          axis=0)
    df = df[adj.columns]
    df.columns = df.columns.to_series().apply(lambda x: "rnaseq." + x)
    print("Final Shapes: ")
    print(df.shape, adj.shape)
    return df, adj


def load_cell_rnaseq(ncols=None, scaling='std', imputing='mean', add_prefix=True,
                     use_landmark_genes=False, use_filtered_genes=False, preprocess_rnaseq=None,
                     embed_feature_source=False, sample_set=None, index_by_sample=False, adj_matrix=None):
    if use_landmark_genes:
        filename = 'combined_rnaseq_data_lincs1000'
    elif use_filtered_genes:
        filename = 'combined_rnaseq_data_filtered'
    else:
        filename = 'combined_rnaseq_data'

    if preprocess_rnaseq and preprocess_rnaseq != 'none':
        scaling = None
        filename += ('_' + preprocess_rnaseq)  # 'source_scale' or 'combat'

    path = get_file(DATA_URL + filename)
    df_cols = pd.read_table(path, engine='c', nrows=0)
    total = df_cols.shape[1] - 1  # remove Sample column
    if 'Cancer_type_id' in df_cols.columns:
        total -= 1
    usecols = None
    if ncols and ncols < total:
        usecols = np.random.choice(total, size=ncols, replace=False)
        usecols = np.append([0], np.add(sorted(usecols), 2))
        df_cols = df_cols.iloc[:, usecols]

    dtype_dict = dict((x, np.float32) for x in df_cols.columns[1:])
    df = pd.read_table(path, engine='c', usecols=usecols, dtype=dtype_dict)
    if 'Cancer_type_id' in df.columns:
        df.drop('Cancer_type_id', axis=1, inplace=True)

    prefixes = df['Sample'].str.extract('^([^.]*)', expand=False).rename('Source')
    sources = prefixes.drop_duplicates().reset_index(drop=True)
    df_source = pd.get_dummies(sources, prefix='rnaseq.source', prefix_sep='.')
    df_source = pd.concat([sources, df_source], axis=1)

    df1 = df['Sample']
    if embed_feature_source:
        df_sample_source = pd.concat([df1, prefixes], axis=1)
        df1 = df_sample_source.merge(df_source, on='Source', how='left').drop('Source', axis=1)
        logger.info('Embedding RNAseq data source into features: %d additional columns', df1.shape[1] - 1)

    df2 = df.drop('Sample', 1)
    if add_prefix:
        df2 = df2.add_prefix('rnaseq.')

    df2 = impute_and_scale(df2, scaling, imputing)

    adj = None
    if adj_matrix is not None:
        df2, adj = align_features(df2, adj_matrix)
    df2.to_hdf("df2_test_needtobe_data.hdf", key="data")
    df = pd.concat([df1, df2], axis=1)

    # scaling needs to be done before subsampling
    if sample_set:
        chosen = df['Sample'].str.startswith(sample_set)
        df = df[chosen].reset_index(drop=True)

    if index_by_sample:
        df = df.set_index('Sample')

    logger.info('Loaded combined RNAseq data: %s', df.shape)

    return df, adj


def select_drugs_with_response_range(df_response, lower=0, upper=0, span=0, lower_median=None, upper_median=None):
    df = df_response.groupby(['Drug1', 'Sample'])['Growth'].agg(['min', 'max', 'median'])
    df['span'] = df['max'].clip(lower=-1, upper=1) - df['min'].clip(lower=-1, upper=1)
    df = df.groupby('Drug1').mean().reset_index().rename(columns={'Drug1': 'Drug'})
    mask = (df['min'] <= lower) & (df['max'] >= upper) & (df['span'] >= span)
    if lower_median:
        mask &= (df['median'] >= lower_median)
    if upper_median:
        mask &= (df['median'] <= upper_median)
    df_sub = df[mask]
    return df_sub


def summarize_response_data(df):
    df_sum = df.groupby('Source').agg({'Growth': 'count', 'Sample': 'nunique',
                                       'Drug1': 'nunique', 'Drug2': 'nunique'})
    df_sum['MedianDose'] = df.groupby('Source').agg({'Dose1': 'median'})
    return df_sum


def assign_partition_groups(df, partition_by='drug_pair'):
    if partition_by == 'cell':
        group = df['Sample']
    elif partition_by == 'drug_pair':
        df_info = load_drug_info()
        id_dict = df_info[['ID', 'PUBCHEM']].drop_duplicates(['ID']).set_index('ID').iloc[:, 0]
        group = df['Drug1'].copy()
        group[(df['Drug2'].notnull()) & (df['Drug1'] <= df['Drug2'])] = df['Drug1'] + ',' + df['Drug2']
        group[(df['Drug2'].notnull()) & (df['Drug1'] > df['Drug2'])] = df['Drug2'] + ',' + df['Drug1']
        group2 = group.map(id_dict)
        mapped = group2.notnull()
        group[mapped] = group2[mapped]
    elif partition_by == 'index':
        group = df.reset_index()['index']
    logger.info('Grouped response data by %s: %d groups', partition_by, group.nunique())
    return group


def dict_compare(d1, d2, ignore=[], expand=False):
    d1_keys = set(d1.keys()) - set(ignore)
    d2_keys = set(d2.keys()) - set(ignore)
    intersect_keys = d1_keys.intersection(d2_keys)
    added = d1_keys - d2_keys
    removed = d2_keys - d1_keys
    modified = set({x: (d1[x], d2[x]) for x in intersect_keys if d1[x] != d2[x]})
    common = set(x for x in intersect_keys if d1[x] == d2[x])
    equal = not (added or removed or modified)
    if expand:
        return equal, added, removed, modified, common
    else:
        return equal, added | removed | modified


def values_or_dataframe(df, contiguous=False, dataframe=False):
    if dataframe:
        return df
    mat = df.values
    if contiguous:
        mat = np.ascontiguousarray(mat)
    return mat


class CombinedDataLoader(object):
    def __init__(self, seed=SEED):
        self.seed = seed

    def load_from_cache(self, cache, params):
        param_fname = '{}.params.json'.format(cache)
        if not os.path.isfile(param_fname):
            logger.warning('Cache parameter file does not exist: %s', param_fname)
            return False
        with open(param_fname) as param_file:
            try:
                cached_params = json.load(param_file)
            except json.JSONDecodeError as e:
                logger.warning('Could not decode parameter file %s: %s', param_fname, e)
                return False
        ignore_keys = ['cache', 'partition_by', 'single']
        equal, diffs = dict_compare(params, cached_params, ignore_keys)
        if not equal:
            logger.warning('Cache parameter mismatch: %s\nSaved: %s\nAttemptd to load: %s', diffs, cached_params,
                           params)
            logger.warning('\nRemove %s to rebuild data cache.\n', param_fname)
            raise ValueError('Could not load from a cache with incompatible keys:', diffs)
        else:
            fname = '{}.pkl'.format(cache)
            if not os.path.isfile(fname):
                logger.warning('Cache file does not exist: %s', fname)
                return False
            with open(fname, 'rb') as f:
                obj = pickle.load(f)
            self.__dict__.update(obj.__dict__)
            logger.info('Loaded data from cache: %s', fname)
            return True
        return False

    def save_to_cache(self, cache, params):
        for k in ['self', 'cache', 'single']:
            if k in params:
                del params[k]
        param_fname = '{}.params.json'.format(cache)
        with open(param_fname, 'w') as param_file:
            json.dump(params, param_file, sort_keys=True)
        fname = '{}.pkl'.format(cache)
        # with open(fname, 'wb') as f:
        # pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
        logger.info('Saved data to cache: %s', fname)

    def partition_data(self, partition_by=None, cv_folds=1, train_split=0.7, val_split=0.2,
                       cell_types=None, by_cell=None, by_drug=None):

        seed = self.seed
        train_sep_sources = self.train_sep_sources
        test_sep_sources = self.test_sep_sources
        df_response = self.df_response

        if not partition_by:
            if by_drug and by_cell:
                partition_by = 'index'
            elif by_drug:
                partition_by = 'cell'
            else:
                partition_by = 'drug_pair'

        if partition_by != self.partition_by:
            df_response = df_response.assign(Group=assign_partition_groups(df_response, partition_by))

        mask = df_response['Source'].isin(train_sep_sources)
        test_mask = df_response['Source'].isin(test_sep_sources)

        if by_drug:
            drug_ids = drug_name_to_ids(by_drug)
            logger.info('Mapped drug IDs for %s: %s', by_drug, drug_ids)
            mask &= (df_response['Drug1'].isin(drug_ids)) & (df_response['Drug2'].isnull())
            test_mask &= (df_response['Drug1'].isin(drug_ids)) & (df_response['Drug2'].isnull())

        if by_cell:
            cell_ids = cell_name_to_ids(by_cell)
            logger.info('Mapped sample IDs for %s: %s', by_cell, cell_ids)
            mask &= (df_response['Sample'].isin(cell_ids))
            test_mask &= (df_response['Sample'].isin(cell_ids))

        if cell_types:
            df_type = load_cell_metadata()
            cell_ids = set()
            for cell_type in cell_types:
                cells = df_type[
                    ~df_type['TUMOR_TYPE'].isnull() & df_type['TUMOR_TYPE'].str.contains(cell_type, case=False)]
                cell_ids |= set(cells['ANL_ID'].tolist())
                logger.info('Mapped sample tissue types for %s: %s', cell_type, set(cells['TUMOR_TYPE'].tolist()))
            mask &= (df_response['Sample'].isin(cell_ids))
            test_mask &= (df_response['Sample'].isin(cell_ids))

        df_group = df_response[mask]['Group'].drop_duplicates().reset_index(drop=True)

        if cv_folds > 1:
            selector = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)
        else:
            selector = ShuffleSplit(n_splits=1, train_size=train_split, test_size=val_split, random_state=seed)

        splits = selector.split(df_group)

        train_indexes = []
        val_indexes = []
        test_indexes = []

        for index, (train_group_index, val_group_index) in enumerate(splits):
            train_groups = set(df_group.values[train_group_index])
            val_groups = set(df_group.values[val_group_index])
            train_index = df_response.index[df_response['Group'].isin(train_groups) & mask]
            val_index = df_response.index[df_response['Group'].isin(val_groups) & mask]
            test_index = df_response.index[
                ~df_response['Group'].isin(train_groups) & ~df_response['Group'].isin(val_groups) & test_mask]

            train_indexes.append(train_index)
            val_indexes.append(val_index)
            test_indexes.append(test_index)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug('CV fold %d: train data = %s, val data = %s, test data = %s', index, train_index.shape[0],
                             val_index.shape[0], test_index.shape[0])
                logger.debug('  train groups (%d): %s', df_response.loc[train_index]['Group'].nunique(),
                             df_response.loc[train_index]['Group'].unique())
                logger.debug('  val groups ({%d}): %s', df_response.loc[val_index]['Group'].nunique(),
                             df_response.loc[val_index]['Group'].unique())
                logger.debug('  test groups ({%d}): %s', df_response.loc[test_index]['Group'].nunique(),
                             df_response.loc[test_index]['Group'].unique())

        self.partition_by = partition_by
        self.cv_folds = cv_folds
        self.train_indexes = train_indexes
        self.val_indexes = val_indexes
        self.test_indexes = test_indexes

    def build_feature_list(self, single=False):
        input_features = collections.OrderedDict()
        feature_shapes = {}

        doses = ['dose1', 'dose2'] if not single else ['dose1']
        for dose in doses:
            input_features[dose] = 'dose'
            feature_shapes['dose'] = (1,)

        if self.encode_response_source:
            input_features['response.source'] = 'response.source'
            feature_shapes['response.source'] = (self.df_source.shape[1] - 1,)

        for fea in self.cell_features:
            feature_type = 'cell.' + fea
            feature_name = 'cell.' + fea
            df_cell = getattr(self, self.cell_df_dict[fea])
            input_features[feature_name] = feature_type
            feature_shapes[feature_type] = (df_cell.shape[1] - 1,)

        drugs = ['drug1', 'drug2'] if not single else ['drug1']
        for drug in drugs:
            for fea in self.drug_features:
                feature_type = 'drug.' + fea
                feature_name = drug + '.' + fea
                df_drug = getattr(self, self.drug_df_dict[fea])
                input_features[feature_name] = feature_type
                feature_shapes[feature_type] = (df_drug.shape[1] - 1,)

        input_dim = sum([np.prod(feature_shapes[x]) for x in input_features.values()])

        self.input_features = input_features
        self.feature_shapes = feature_shapes
        self.input_dim = input_dim

        logger.info('Input features shapes:')
        for k, v in self.input_features.items():
            logger.info('  {}: {}'.format(k, self.feature_shapes[v]))
        logger.info('Total input dimensions: {}'.format(self.input_dim))

    def load(self, cache=None, ncols=None, scaling='std', dropna=None,
             embed_feature_source=True, encode_response_source=True,
             cell_features=['rnaseq'], drug_features=['descriptors', 'fingerprints'],
             drug_lower_response=1, drug_upper_response=-1, drug_response_span=0,
             drug_median_response_min=-1, drug_median_response_max=1,
             use_landmark_genes=False, use_filtered_genes=False,
             preprocess_rnaseq=None, single=False,
             # train_sources=['GDSC', 'CTRP', 'ALMANAC', 'NCI60'],
             train_sources=['GDSC', 'CTRP', 'ALMANAC'],
             # val_sources='train',
             # test_sources=['CCLE', 'gCSI'],
             test_sources=['train'],
             partition_by='drug_pair',
             use_genemania_adj_matrix=False,
             genemania_file_path=None):

        params = locals().copy()
        del params['self']

        if not cell_features or 'none' in [x.lower() for x in cell_features]:
            cell_features = []

        if not drug_features or 'none' in [x.lower() for x in drug_features]:
            drug_features = []

        if cache and self.load_from_cache(cache, params):
            self.build_feature_list(single=single)
            return

        logger.info('Loading data from scratch ...')

        # ncols=None
        # scaling='std'
        # embed_feature_source=True
        # dropna=None
        # cell_features=['rnaseq']
        # drug_features=['descriptors', 'fingerprints']
        # # train_sources=['CTRP']
        # train_sources=['GDSC', 'CTRP', 'ALMANAC']
        # # train_sources=['GDSC', 'CTRP', 'ALMANAC', 'NCI60']
        # test_sources=['CCLE', 'gCSI']
        # val_sources='train'

        # use_landmark_genes=True

        # drug_lower_response=-0.4
        # drug_upper_response=0.4
        # drug_response_span=1.2

        # drug_lower_response=0
        # drug_upper_response=0
        # drug_response_span=1

        # drug_lower_response=1
        # drug_upper_response=-1
        # drug_response_span=0

        df_response = load_combined_dose_response()
        if logger.isEnabledFor(logging.INFO):
            logger.info('Summary of combined dose response by source:')
            logger.info(summarize_response_data(df_response))

        all_sources = df_response['Source'].unique()
        print("Want to use %s" % train_sources )
        print ("SOURCES:")
        print (np.unique(df_response['Source'], return_counts=True))
        df_source = encode_sources(all_sources)

        if 'all' in train_sources:
            train_sources = all_sources
        if 'all' in test_sources:
            test_sources = all_sources
        elif 'train' in test_sources:
            test_sources = train_sources

        train_sep_sources = [x for x in all_sources for y in train_sources if x.startswith(y)]
        test_sep_sources = [x for x in all_sources for y in test_sources if x.startswith(y)]

        ids1 = df_response[['Drug1']].drop_duplicates().rename(columns={'Drug1': 'Drug'})
        ids2 = df_response[['Drug2']].drop_duplicates().rename(columns={'Drug2': 'Drug'})
        df_drugs_with_response = pd.concat([ids1, ids2]).drop_duplicates().dropna().reset_index(drop=True)
        df_cells_with_response = df_response[['Sample']].drop_duplicates().reset_index(drop=True)
        logger.info('Combined raw dose response data has %d unique samples and %d unique drugs',
                    df_cells_with_response.shape[0], df_drugs_with_response.shape[0])

        logger.info(
            'Limiting drugs to those with response min <= %g, max >= %g, span >= %g, median_min <= %g, median_max >= %g ...',
            drug_lower_response, drug_upper_response, drug_response_span, drug_median_response_min,
            drug_median_response_max)
        df_selected_drugs = select_drugs_with_response_range(df_response, span=drug_response_span,
                                                             lower=drug_lower_response, upper=drug_upper_response,
                                                             lower_median=drug_median_response_min,
                                                             upper_median=drug_median_response_max)
        logger.info('Selected %d drugs from %d', df_selected_drugs.shape[0], df_response['Drug1'].nunique())

        self.adj = None
        if use_genemania_adj_matrix:
            self.adj = load_genemania_adj_matrix(genemania_file_path)
        for fea in cell_features:
            fea = fea.lower()
            if fea == 'rnaseq' or fea == 'expression':
                df_cell_rnaseq, self.adj = load_cell_rnaseq(ncols=ncols, scaling=scaling,
                                                            use_landmark_genes=use_landmark_genes,
                                                            use_filtered_genes=use_filtered_genes,
                                                            preprocess_rnaseq=preprocess_rnaseq,
                                                            embed_feature_source=embed_feature_source,
                                                            adj_matrix=self.adj)

        for fea in drug_features:
            fea = fea.lower()
            if fea == 'descriptors':
                df_drug_desc = load_drug_descriptors(ncols=ncols, scaling=scaling, dropna=dropna)
            elif fea == 'fingerprints':
                df_drug_fp = load_drug_fingerprints(ncols=ncols, scaling=scaling, dropna=dropna)

        # df_drug_desc, df_drug_fp = load_drug_data(ncols=ncols, scaling=scaling, dropna=dropna)

        cell_df_dict = {'rnaseq': 'df_cell_rnaseq'}

        drug_df_dict = {'descriptors': 'df_drug_desc',
                        'fingerprints': 'df_drug_fp'}

        # df_cell_ids = df_cell_rnaseq[['Sample']].drop_duplicates()
        # df_drug_ids = pd.concat([df_drug_desc[['Drug']], df_drug_fp[['Drug']]]).drop_duplicates()

        logger.info('Filtering drug response data...')

        df_cell_ids = df_cells_with_response
        for fea in cell_features:
            df_cell = locals()[cell_df_dict[fea]]
            df_cell_ids = df_cell_ids.merge(df_cell[['Sample']]).drop_duplicates()

        logger.info('  %d molecular samples with feature and response data', df_cell_ids.shape[0])

        df_drug_ids = df_drugs_with_response
        for fea in drug_features:
            df_drug = locals()[drug_df_dict[fea]]
            df_drug_ids = df_drug_ids.merge(df_drug[['Drug']]).drop_duplicates()

        df_drug_ids = df_drug_ids.merge(df_selected_drugs).drop_duplicates()
        logger.info('  %d selected drugs with feature and response data', df_drug_ids.shape[0])

        df_response = df_response[df_response['Sample'].isin(df_cell_ids['Sample']) &
                                  df_response['Drug1'].isin(df_drug_ids['Drug']) &
                                  (df_response['Drug2'].isin(df_drug_ids['Drug']) | df_response['Drug2'].isnull())]

        df_response = df_response[df_response['Source'].isin(train_sep_sources + test_sep_sources)]

        df_response.reset_index(drop=True, inplace=True)

        if logger.isEnabledFor(logging.INFO):
            logger.info('Summary of filtered dose response by source:')
            logger.info(summarize_response_data(df_response))

        df_response = df_response.assign(Group=assign_partition_groups(df_response, partition_by))

        self.cell_features = cell_features
        self.drug_features = drug_features
        self.cell_df_dict = cell_df_dict
        self.drug_df_dict = drug_df_dict
        self.df_source = df_source
        self.df_response = df_response
        self.embed_feature_source = embed_feature_source
        self.encode_response_source = encode_response_source
        self.all_sources = all_sources
        self.train_sources = train_sources
        self.test_sources = test_sources
        self.train_sep_sources = train_sep_sources
        self.test_sep_sources = test_sep_sources
        self.partition_by = partition_by

        for var in (list(drug_df_dict.values()) + list(cell_df_dict.values())):
            value = locals().get(var)
            if value is not None:
                setattr(self, var, value)

        self.build_feature_list(single=single)

        if cache:
            self.save_to_cache(cache, params)


class CombinedDataGenerator(object):
    """Generate training, validation or testing batches from loaded data
    """

    def __init__(self, data, partition='train', fold=0, source=None, batch_size=32, shuffle=True):
        self.data = data
        self.partition = partition
        self.batch_size = batch_size

        if partition == 'train':
            index = data.train_indexes[fold]
        elif partition == 'val':
            index = data.val_indexes[fold]
        else:
            index = data.test_indexes[fold]

        if source:
            df = data.df_response[['Source']].iloc[index, :]
            index = df.index[df['Source'] == source]

        if shuffle:
            index = np.random.permutation(index)
        # index = index[:len(index)//10]

        self.index = index
        self.index_cycle = cycle(index)
        self.size = len(index)
        self.steps = np.ceil(self.size / batch_size)
        # self.steps = np.ceil(self.size / batch_size / 100)

    def reset(self):
        self.index_cycle = cycle(self.index)

    def get_response(self, copy=False):
        df = self.data.df_response.iloc[self.index, :].drop(['Group'], axis=1)
        return df.copy() if copy else df

    def get_slice(self, size=None, contiguous=True, single=False, dataframe=False):
        size = size or self.size

        index = list(islice(self.index_cycle, size))
        df_orig = self.data.df_response.iloc[index, :]
        df = df_orig.copy()

        df['Swap'] = np.random.choice([True, False], df.shape[0])
        df['DoseSplit'] = np.random.uniform(0.001, 0.999, df.shape[0])

        swap = df_orig['Drug2'].notnull() & df['Swap']
        df.loc[swap, 'Drug1'] = df_orig.loc[swap, 'Drug2']
        df.loc[swap, 'Dose1'] = df_orig.loc[swap, 'Dose2']
        df.loc[swap, 'Drug2'] = df_orig.loc[swap, 'Drug1']
        df.loc[swap, 'Dose2'] = df_orig.loc[swap, 'Dose1']

        split = df_orig['Drug2'].isnull()
        if not single:
            df.loc[split, 'Drug2'] = df_orig.loc[split, 'Drug1']
            df.loc[split, 'Dose1'] = df_orig.loc[split, 'Dose1'] - np.log10(df.loc[split, 'DoseSplit'])
            df.loc[split, 'Dose2'] = df_orig.loc[split, 'Dose1'] - np.log10(1 - df.loc[split, 'DoseSplit'])

        if dataframe:
            cols = ['Growth', 'Sample', 'Drug1', 'Drug2'] if not single else ['Growth', 'Sample', 'Drug1']
            y = df[cols].reset_index(drop=True)
        else:
            y = values_or_dataframe(df['Growth'], contiguous, dataframe)

        x_list = []

        doses = ['Dose1', 'Dose2'] if not single else ['Dose1']
        for dose in doses:
            x = values_or_dataframe(df[[dose]].reset_index(drop=True), contiguous, dataframe)
            x_list.append(x)

        if self.data.encode_response_source:
            df_x = pd.merge(df[['Source']], self.data.df_source, on='Source', how='left')
            df_x.drop(['Source'], axis=1, inplace=True)
            x = values_or_dataframe(df_x, contiguous, dataframe)
            x_list.append(x)

        for fea in self.data.cell_features:
            df_cell = getattr(self.data, self.data.cell_df_dict[fea])
            df_x = pd.merge(df[['Sample']], df_cell, on='Sample', how='left')
            df_x.drop(['Sample'], axis=1, inplace=True)
            x = values_or_dataframe(df_x, contiguous, dataframe)
            x_list.append(x)

        drugs = ['Drug1', 'Drug2'] if not single else ['Drug1']
        for drug in drugs:
            for fea in self.data.drug_features:
                df_drug = getattr(self.data, self.data.drug_df_dict[fea])
                df_x = pd.merge(df[[drug]], df_drug, left_on=drug, right_on='Drug', how='left')
                df_x.drop([drug, 'Drug'], axis=1, inplace=True)
                if dataframe and not single:
                    df_x = df_x.add_prefix(drug + '.')
                x = values_or_dataframe(df_x, contiguous, dataframe)
                x_list.append(x)

        # print(x_list, y)
        return x_list, y

    def flow(self, single=False):
        while 1:
            x_list, y = self.get_slice(self.batch_size, single=single)
            yield x_list, y


def test_generator(loader):
    gen = CombinedDataGenerator(loader).flow()
    x_list, y = next(gen)
    print('x shapes:')
    for x in x_list:
        print(x.shape)
    print('y shape:')
    print(y.shape)
