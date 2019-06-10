from __future__ import print_function

import collections
import json
import logging
import os
import pickle

import pandas as pd
import numpy as np

from sklearn.model_selection import ShuffleSplit, KFold

import cellline_data
import drug_data
import response_data

from uno import loggerUno as logger
from uno import dict_compare

SEED = 2019

def encode_sources(sources):
    df = pd.get_dummies(sources, prefix='source', prefix_sep='.')
    df['Source'] = sources
    source_l1 = df['Source'].str.extract('^(\S+)\.', expand=False)
    df1 = pd.get_dummies(source_l1, prefix='source.L1', prefix_sep='.')
    df = pd.concat([df1, df], axis=1)
    df = df.set_index('Source').reset_index()
    return df

def read_set_from_file(path):
    if path:
        with open(path, 'r') as f:
            text = f.read().strip()
            subset = text.split()
    else:
        subset = None
    return subset


def assign_partition_groups(df, partition_by='drug_pair'):
    if partition_by == 'cell':
        group = df['Sample']
    elif partition_by == 'drug_pair':
        df_info = drug_data.load_drug_info()
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


class CombinedDataLoader(object):
    def __init__(self, seed=SEED):
        self.seed = seed
        self.test_indexes = [[]]

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
            logger.warning('Cache parameter mismatch: %s\nSaved: %s\nAttemptd to load: %s', diffs, cached_params, params)
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
        with open(fname, 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
        logger.info('Saved data to cache: %s', fname)

    def partition_data(self, partition_by=None, cv_folds=1, train_split=0.7, val_split=0.2,
                       cell_types=None, by_cell=None, by_drug=None,
                       cell_subset_path=None, drug_subset_path=None,
                       exclude_cells=[], exclude_drugs=[], exclude_indices=[]):

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


       # Exclude specified cells / drugs / indices
        if exclude_cells != []:
            df_response = df_response[~df_response['Sample'].isin(exclude_cells)]
        if exclude_drugs != []:
            if np.isin('Drug', df_response.columns.values):
                df_response = df_response[~df_response['Drug1'].isin(exclude_drugs)]
            else:
                df_response = df_response[~df_response['Drug1'].isin(exclude_drugs) & ~df_response['Drug2'].isin(exclude_drugs)]
        if exclude_indices != []:
            df_response = df_response.drop(exclude_indices, axis=0)
            logger.info('Excluding indices specified')

        if partition_by != self.partition_by:
            df_response = df_response.assign(Group = assign_partition_groups(df_response, partition_by))

        mask = df_response['Source'].isin(train_sep_sources)
        test_mask = df_response['Source'].isin(test_sep_sources)

        if by_drug:
            drug_ids = drug_data.drug_name_to_ids(by_drug)
            logger.info('Mapped drug IDs for %s: %s', by_drug, drug_ids)
            mask &= (df_response['Drug1'].isin(drug_ids)) & (df_response['Drug2'].isnull())
            test_mask &= (df_response['Drug1'].isin(drug_ids)) & (df_response['Drug2'].isnull())

        if by_cell:
            cell_ids = cellline_data.cell_name_to_ids(by_cell)
            logger.info('Mapped sample IDs for %s: %s', by_cell, cell_ids)
            mask &= (df_response['Sample'].isin(cell_ids))
            test_mask &= (df_response['Sample'].isin(cell_ids))

        if cell_subset_path:
            cell_subset = read_set_from_file(cell_subset_path)
            mask &= (df_response['Sample'].isin(cell_subset))
            test_mask &= (df_response['Sample'].isin(cell_subset))

        if drug_subset_path:
            drug_subset = read_set_from_file(drug_subset_path)
            mask &= (df_response['Drug1'].isin(drug_subset)) & ((df_response['Drug2'].isnull()) | (df_response['Drug2'].isin(drug_subset)))
            test_mask &= (df_response['Drug1'].isin(drug_subset)) & ((df_response['Drug2'].isnull()) | (df_response['Drug2'].isin(drug_subset)))

        if cell_types:
            df_type = cellline_data.load_cell_metadata()
            cell_ids = set()
            for cell_type in cell_types:
                cells = df_type[~df_type['TUMOR_TYPE'].isnull() & df_type['TUMOR_TYPE'].str.contains(cell_type, case=False)]
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
            test_index = df_response.index[~df_response['Group'].isin(train_groups) & ~df_response['Group'].isin(val_groups) & test_mask]

            train_indexes.append(train_index)
            val_indexes.append(val_index)
            test_indexes.append(test_index)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug('CV fold %d: train data = %s, val data = %s, test data = %s', index, train_index.shape[0], val_index.shape[0], test_index.shape[0])
                logger.debug('  train groups (%d): %s', df_response.loc[train_index]['Group'].nunique(), df_response.loc[train_index]['Group'].unique())
                logger.debug('  val groups ({%d}): %s', df_response.loc[val_index]['Group'].nunique(), df_response.loc[val_index]['Group'].unique())
                logger.debug('  test groups ({%d}): %s', df_response.loc[test_index]['Group'].nunique(), df_response.loc[test_index]['Group'].unique())


        self.partition_by = partition_by
        self.cv_folds = cv_folds
        self.train_indexes = train_indexes
        self.val_indexes = val_indexes
        self.test_indexes = test_indexes

    def build_feature_list(self, single=False):
        input_features = collections.OrderedDict()
        feature_shapes = collections.OrderedDict()

        if not self.agg_dose:
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
             agg_dose=None, embed_feature_source=True, encode_response_source=True,
             cell_features=['rnaseq'], drug_features=['descriptors', 'fingerprints'],
             cell_feature_subset_path=None, drug_feature_subset_path=None,
             drug_lower_response=1, drug_upper_response=-1, drug_response_span=0,
             drug_median_response_min=-1, drug_median_response_max=1,
             use_landmark_genes=False, use_filtered_genes=False,
             preprocess_rnaseq=None, single=False,
             # train_sources=['GDSC', 'CTRP', 'ALMANAC', 'NCI60'],
             train_sources=['GDSC', 'CTRP', 'ALMANAC'],
             # val_sources='train',
             # test_sources=['CCLE', 'gCSI'],
             test_sources=['train'],
             partition_by='drug_pair'):

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

        if agg_dose:
            df_response = response_data.load_aggregated_single_response(target=agg_dose, combo_format=True)
        else:
            df_response = response_data.load_combined_dose_response()

        if logger.isEnabledFor(logging.INFO):
            logger.info('Summary of combined dose response by source:')
            logger.info(response_data.summarize_response_data(df_response, target=agg_dose))

        all_sources = df_response['Source'].unique()
        df_source = encode_sources(all_sources)

        if 'all' in train_sources:
            train_sources = all_sources
        if 'all' in test_sources:
            test_sources = all_sources
        elif 'train' in test_sources:
            test_sources = train_sources

        train_sep_sources = [x for x in all_sources for y in train_sources if x.startswith(y)]
        test_sep_sources = [x for x in all_sources for y in test_sources if x.startswith(y)]

        ids1 = df_response[['Drug1']].drop_duplicates().rename(columns={'Drug1':'Drug'})
        ids2 = df_response[['Drug2']].drop_duplicates().rename(columns={'Drug2':'Drug'})
        df_drugs_with_response = pd.concat([ids1, ids2]).drop_duplicates().dropna().reset_index(drop=True)
        df_cells_with_response = df_response[['Sample']].drop_duplicates().reset_index(drop=True)
        logger.info('Combined raw dose response data has %d unique samples and %d unique drugs', df_cells_with_response.shape[0], df_drugs_with_response.shape[0])

        if agg_dose:
            df_selected_drugs = None
        else:
            logger.info('Limiting drugs to those with response min <= %g, max >= %g, span >= %g, median_min <= %g, median_max >= %g ...', drug_lower_response, drug_upper_response, drug_response_span, drug_median_response_min, drug_median_response_max)
            df_selected_drugs = response_data.select_drugs_with_response_range(df_response, span=drug_response_span, lower=drug_lower_response, upper=drug_upper_response, lower_median=drug_median_response_min, upper_median=drug_median_response_max)
            logger.info('Selected %d drugs from %d', df_selected_drugs.shape[0], df_response['Drug1'].nunique())


        cell_feature_subset = read_set_from_file(cell_feature_subset_path)
        drug_feature_subset = read_set_from_file(drug_feature_subset_path)

        for fea in cell_features:
            fea = fea.lower()
            if fea == 'rnaseq' or fea == 'expression':
                df_cell_rnaseq = cellline_data.load_cell_rnaseq(ncols=ncols, scaling=scaling, use_landmark_genes=use_landmark_genes, use_filtered_genes=use_filtered_genes, feature_subset=cell_feature_subset, preprocess_rnaseq=preprocess_rnaseq, embed_feature_source=embed_feature_source)

        for fea in drug_features:
            fea = fea.lower()
            if fea == 'descriptors':
                df_drug_desc = drug_data.load_drug_descriptors(ncols=ncols, scaling=scaling, dropna=dropna, feature_subset=drug_feature_subset)
            elif fea == 'fingerprints':
                df_drug_fp = drug_data.load_drug_fingerprints(ncols=ncols, scaling=scaling, dropna=dropna, feature_subset=drug_feature_subset)

        # df_drug_desc, df_drug_fp = drug_data.load_drug_data(ncols=ncols, scaling=scaling, dropna=dropna)

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

        if df_selected_drugs is not None:
            df_drug_ids = df_drug_ids.merge(df_selected_drugs).drop_duplicates()
        logger.info('  %d selected drugs with feature and response data', df_drug_ids.shape[0])

        df_response = df_response[df_response['Sample'].isin(df_cell_ids['Sample']) &
                                  df_response['Drug1'].isin(df_drug_ids['Drug']) &
                                  (df_response['Drug2'].isin(df_drug_ids['Drug']) | df_response['Drug2'].isnull())]

        df_response = df_response[df_response['Source'].isin(train_sep_sources + test_sep_sources)]

        df_response.reset_index(drop=True, inplace=True)

        if logger.isEnabledFor(logging.INFO):
            logger.info('Summary of filtered dose response by source:')
            logger.info(response_data.summarize_response_data(df_response, target=agg_dose))

        df_response = df_response.assign(Group = assign_partition_groups(df_response, partition_by))

        self.agg_dose = agg_dose
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

        for var in (list(drug_df_dict.values()) +  list(cell_df_dict.values())):
            value = locals().get(var)
            if value is not None:
                setattr(self, var, value)

        self.build_feature_list(single=single)

        if cache:
            self.save_to_cache(cache, params)


    def get_cells_in_val(self):
    
        val_cell_ids = list(set(self.df_response.loc[self.val_indexes[0]]['Sample'].values))
    
        return val_cell_ids


    def get_drugs_in_val(self):
    
        if np.isin('Drug', self.df_response.columns.values):
            val_drug_ids = list(set(self.df_response.loc[self.val_indexes[0]]['Drug'].values))
        else:
            val_drug_ids = list(set(self.df_response.loc[self.val_indexes[0]]['Drug1'].values))
    
        return val_drug_ids


    def get_index_in_val(self):
    
        val_indices = list(set(self.val_indexes[0]))
    
        return val_indices


