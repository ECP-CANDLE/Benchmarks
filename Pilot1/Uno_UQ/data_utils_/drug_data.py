
import pandas as pd
import numpy as np

import candle_keras as candle

from uno import get_file_p1 as get_file
from uno import loggerUno as logger
from uno import DATA_URL


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
    df2 = candle.drop_impute_and_scale_dataframe(df2, scaling=scaling, imputing=imputing, dropna=dropna)
    if add_prefix:
        df2 = df2.add_prefix('dragon7.')
    df_desc = pd.concat([df1, df2], axis=1)

    df_fp = pd.concat([df_fp, df_fp2]).reset_index(drop=True)
    df1 = pd.DataFrame(df_fp.loc[:, 'Drug'])
    df2 = df_fp.drop('Drug', 1)
    df2 = candle.drop_impute_and_scale_dataframe(df2, scaling=None, imputing=imputing, dropna=dropna)
    if add_prefix:
        df2 = df2.add_prefix('dragon7.')
    df_fp = pd.concat([df1, df2], axis=1)

    logger.info('Loaded combined dragon7 drug descriptors: %s', df_desc.shape)
    logger.info('Loaded combined dragon7 drug fingerprints: %s', df_fp.shape)

    return df_desc, df_fp


def load_drug_descriptors(ncols=None, scaling='std', imputing='mean', dropna=None, add_prefix=True, feature_subset=None):
    df_info = load_drug_info()
    df_info['Drug'] = df_info['PUBCHEM']

    df_desc = load_drug_set_descriptors(drug_set='Combined_PubChem', ncols=ncols)
    df_desc = pd.merge(df_info[['ID', 'Drug']], df_desc, on='Drug').drop('Drug', 1).rename(columns={'ID': 'Drug'})

    df_desc2 = load_drug_set_descriptors(drug_set='NCI60', usecols=df_desc.columns.tolist() if ncols else None)

    df_desc = pd.concat([df_desc, df_desc2]).reset_index(drop=True)
    df1 = pd.DataFrame(df_desc.loc[:, 'Drug'])
    df2 = df_desc.drop('Drug', 1)
    if add_prefix:
        df2 = df2.add_prefix('dragon7.')
    if feature_subset:
        df2 = df2[[x for x in df2.columns if x in feature_subset]]
    df2 = candle.drop_impute_and_scale_dataframe(df2, scaling=scaling, imputing=imputing, dropna=dropna)
    df_desc = pd.concat([df1, df2], axis=1)

    logger.info('Loaded combined dragon7 drug descriptors: %s', df_desc.shape)

    return df_desc


def load_drug_fingerprints(ncols=None, scaling='std', imputing='mean', dropna=None, add_prefix=True, feature_subset=None):
    df_info = load_drug_info()
    df_info['Drug'] = df_info['PUBCHEM']

    df_fp = load_drug_set_fingerprints(drug_set='Combined_PubChem', ncols=ncols)
    df_fp = pd.merge(df_info[['ID', 'Drug']], df_fp, on='Drug').drop('Drug', 1).rename(columns={'ID': 'Drug'})

    df_fp2 = load_drug_set_fingerprints(drug_set='NCI60', usecols=df_fp.columns.tolist() if ncols else None)

    df_fp = pd.concat([df_fp, df_fp2]).reset_index(drop=True)
    df1 = pd.DataFrame(df_fp.loc[:, 'Drug'])
    df2 = df_fp.drop('Drug', 1)
    if add_prefix:
        df2 = df2.add_prefix('dragon7.')
    if feature_subset:
        df2 = df2[[x for x in df2.columns if x in feature_subset]]
    df2 = candle.drop_impute_and_scale_dataframe(df2, scaling=None, imputing=imputing, dropna=dropna)
    df_fp = pd.concat([df1, df2], axis=1)

    logger.info('Loaded combined dragon7 drug fingerprints: %s', df_fp.shape)

    return df_fp


def load_drug_info():
    path = get_file(DATA_URL + 'drug_info')
    df = pd.read_csv(path, sep='\t', dtype=object)
    df['PUBCHEM'] = 'PubChem.CID.' + df['PUBCHEM']
    return df


def drug_name_to_ids(name, source=None):
    df1 = load_drug_info()
    path = get_file(DATA_URL + 'NCI_IOA_AOA_drugs')
    df2 = pd.read_csv(path, sep='\t', dtype=str)
    df2['NSC'] = 'NSC.' + df2['NSC']
    hits1 = candle.lookup(df1, name, 'ID', ['ID', 'NAME', 'CLEAN_NAME', 'PUBCHEM'])
    hits2 = candle.lookup(df2, name, 'NSC', ['NSC', 'Generic Name', 'Preffered Name'])
    hits = hits1 + hits2
    if source:
        hits = [x for x in hits if x.startswith(source.upper()+'.')]
    return hits


def load_drug_set_descriptors(drug_set='Combined_PubChem', ncols=None, usecols=None,
                              scaling=None, imputing=None, add_prefix=False):
    path = get_file(DATA_URL + '{}_dragon7_descriptors.tsv'.format(drug_set))

    df_cols = pd.read_csv(path, engine='c', sep='\t', nrows=0)
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
    df = pd.read_csv(path, engine='c', sep='\t', usecols=usecols, dtype=dtype_dict,
                     na_values=['na', '-', ''])

    df1 = pd.DataFrame(df.loc[:, 'NAME'])
    df1.rename(columns={'NAME': 'Drug'}, inplace=True)

    df2 = df.drop('NAME', 1)
    if add_prefix:
        df2 = df2.add_prefix('dragon7.')

    df2 = candle.drop_impute_and_scale_dataframe(df2, scaling, imputing, dropna=None)

    df = pd.concat([df1, df2], axis=1)
    return df


def load_drug_set_fingerprints(drug_set='Combined_PubChem', ncols=None, usecols=None,
                               scaling=None, imputing=None, add_prefix=False):
    fps = ['PFP', 'ECFP']
    usecols_all = usecols
    df_merged = None
    for fp in fps:
        path = get_file(DATA_URL + '{}_dragon7_{}.tsv'.format(drug_set, fp))
        df_cols = pd.read_csv(path, engine='c', sep='\t', nrows=0, skiprows=1, header=None)
        total = df_cols.shape[1] - 1
        if usecols_all is not None:
            usecols = [x.replace(fp+'.', '') for x in usecols_all]
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
        df = pd.read_csv(path, engine='c', sep='\t', skiprows=1, header=None,
                         usecols=usecols, dtype=dtype_dict)
        df.columns = ['{}.{}'.format(fp, x) for x in df.columns]

        col1 = '{}.0'.format(fp)
        df1 = pd.DataFrame(df.loc[:, col1])
        df1.rename(columns={col1: 'Drug'}, inplace=True)

        df2 = df.drop(col1, 1)
        if add_prefix:
            df2 = df2.add_prefix('dragon7.')

        df2 = candle.drop_impute_and_scale_dataframe(df2, scaling, imputing, dropna=None)

        df = pd.concat([df1, df2], axis=1)

        df_merged = df if df_merged is None else df_merged.merge(df)

    return df_merged
