
import pandas as pd
import numpy as np

from uno import get_file_p1 as get_file
from uno import loggerUno as logger
from uno import DATA_URL

global_cache = {}

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
                         na_values=['na','-',''],
                         usecols=['CELLNAME', 'NSC1', 'CONC1', 'NSC2', 'CONC2',
                                  'PERCENTGROWTH', 'VALID', 'SCREENER', 'STUDY'],
                         # nrows=10000,
                         dtype={'CELLNAME': str, 'NSC1': str, 'NSC2': str,
                                'CONC1': np.float32, 'CONC2': np.float32,
                                'PERCENTGROWTH':np.float32, 'VALID': str,
                                'SCREENER': str, 'STUDY': str},
                         error_bad_lines=False, warn_bad_lines=True)
        global_cache[path] = df

    df = df[df['VALID'] == 'Y']

    df['SOURCE'] = 'ALMANAC.' + df['SCREENER']

    cellmap_path = get_file(DATA_URL + 'NCI60_CELLNAME_to_Combo.txt')
    df_cellmap = pd.read_csv(cellmap_path, sep='\t')
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


def load_aggregated_single_response(target='AUC', min_r2_fit=0.3, max_ec50_se=3, combo_format=False, rename=True):
    path = get_file(DATA_URL + 'combined_single_response_agg')

    df = global_cache.get(path)
    if df is None:
        df = pd.read_csv(path, engine='c', sep='\t',
                         dtype={'SOURCE': str, 'CELL': str, 'DRUG': str, 'STUDY': str,
                                'AUC': np.float32, 'IC50': np.float32,
                                'EC50': np.float32, 'EC50se': np.float32,
                                'R2fit': np.float32, 'Einf': np.float32,
                                'HS': np.float32, 'AAC1': np.float32,
                                'AUC1': np.float32, 'DSS1': np.float32})
        global_cache[path] = df

    total = len(df)

    df = df[(df['R2fit'] >= min_r2_fit) & (df['EC50se'] <= max_ec50_se)]
    df = df[['SOURCE', 'CELL', 'DRUG', target, 'STUDY']]
    df = df[~df[target].isnull()]

    logger.info('Loaded %d dose independent response samples (filtered by EC50se <= %f & R2fit >=%f from a total of %d).', len(df), max_ec50_se, min_r2_fit, total)

    if combo_format:
        df = df.rename(columns={'DRUG': 'DRUG1'})
        df['DRUG2'] = np.nan
        df['DRUG2'] = df['DRUG2'].astype(object)
        df = df[['SOURCE', 'CELL', 'DRUG1', 'DRUG2', target, 'STUDY']]
        if rename:
            df = df.rename(columns={'SOURCE': 'Source', 'CELL': 'Sample',
                                    'DRUG1': 'Drug1', 'DRUG2': 'Drug2', 'STUDY': 'Study'})
    else:
        if rename:
            df = df.rename(columns={'SOURCE': 'Source', 'CELL': 'Sample',
                                    'DRUG': 'Drug', 'STUDY': 'Study'})

    return df



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


def summarize_response_data(df, target=None):
    target = target or 'Growth'
    df_sum = df.groupby('Source').agg({target: 'count', 'Sample': 'nunique',
                                       'Drug1': 'nunique', 'Drug2': 'nunique'})
    if 'Dose1' in df_sum:
        df_sum['MedianDose'] = df.groupby('Source').agg({'Dose1': 'median'})
    return df_sum




