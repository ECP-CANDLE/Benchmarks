
import pandas as pd
import numpy as np

import candle_keras as candle

from uno import get_file_p1 as get_file
from uno import loggerUno as logger
from uno import DATA_URL


def load_cell_metadata():
    path = get_file(DATA_URL + 'cl_metadata')
    df = pd.read_csv(path, sep='\t')
    return df


def cell_name_to_ids(name, source=None):
    path = get_file(DATA_URL + 'NCI60_CELLNAME_to_Combo.txt')
    df1 = pd.read_csv(path, sep='\t')
    hits1 = candle.lookup(df1, name, 'NCI60.ID', ['NCI60.ID', 'CELLNAME', 'Name'], match='contains')
    path = get_file(DATA_URL + 'cl_mapping')
    df2 = pd.read_csv(path, sep='\t', header=None)
    hits2 = candle.lookup(df2, name, [0, 1], [0, 1], match='contains')
    hits = hits1 + hits2
    if source:
        hits = [x for x in hits if x.startswith(source.upper()+'.')]
    return hits


def load_cell_rnaseq(ncols=None, scaling='std', imputing='mean', add_prefix=True,
                     use_landmark_genes=False, use_filtered_genes=False,
                     feature_subset=None, preprocess_rnaseq=None,
                     embed_feature_source=False, sample_set=None, index_by_sample=False):

    if use_landmark_genes:
        filename =  'combined_rnaseq_data_lincs1000'
    elif use_filtered_genes:
        filename = 'combined_rnaseq_data_filtered'
    else:
        filename = 'combined_rnaseq_data'

    if preprocess_rnaseq and preprocess_rnaseq != 'none':
        scaling = None
        filename += ('_' + preprocess_rnaseq)  # 'source_scale' or 'combat'

    path = get_file(DATA_URL + filename)
    df_cols = pd.read_csv(path, engine='c', sep='\t', nrows=0)
    total = df_cols.shape[1] - 1  # remove Sample column
    if 'Cancer_type_id' in df_cols.columns:
        total -= 1
    usecols = None
    if ncols and ncols < total:
        usecols = np.random.choice(total, size=ncols, replace=False)
        usecols = np.append([0], np.add(sorted(usecols), 2))
        df_cols = df_cols.iloc[:, usecols]
    if feature_subset:
        with_prefix = lambda x: 'rnaseq.'+x if add_prefix else x
        usecols = [0] + [i for i, c in enumerate(df_cols.columns) if with_prefix(c) in feature_subset]
        df_cols = df_cols.iloc[:, usecols]

    dtype_dict = dict((x, np.float32) for x in df_cols.columns[1:])
    df = pd.read_csv(path, engine='c', sep='\t', usecols=usecols, dtype=dtype_dict)
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
        logger.info('Embedding RNAseq data source into features: %d additional columns', df1.shape[1]-1)

    df2 = df.drop('Sample', 1)
    if add_prefix:
        df2 = df2.add_prefix('rnaseq.')

    df2 = candle.drop_impute_and_scale_dataframe(df2, scaling, imputing)

    df = pd.concat([df1, df2], axis=1)

    # scaling needs to be done before subsampling
    if sample_set:
        chosen = df['Sample'].str.startswith(sample_set)
        df = df[chosen].reset_index(drop=True)

    if index_by_sample:
        df = df.set_index('Sample')

    logger.info('Loaded combined RNAseq data: %s', df.shape)

    return df

