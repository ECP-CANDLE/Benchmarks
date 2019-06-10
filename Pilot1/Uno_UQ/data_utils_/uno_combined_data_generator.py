
from itertools import cycle, islice

import numpy as np
import pandas as pd

from keras.utils import Sequence

def values_or_dataframe(df, contiguous=False, dataframe=False):
    if dataframe:
        return df
    mat = df.values
    if contiguous:
        mat = np.ascontiguousarray(mat)
    return mat


class CombinedDataGenerator(Sequence):#object):
    """Generate training, validation or testing batches from loaded data
    """
#    def __init__(self, data, partition='train', fold=0, source=None, batch_size=32, shuffle=True):
    def __init__(self, data, partition='train', fold=0, source=None, batch_size=32, shuffle=True, single=False, rank=0, total_ranks=1):

        self.data = data
        self.partition = partition
        self.batch_size = batch_size
        self.single = single

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

        # sharing by rank
        samples_per_rank = len(index) // total_ranks
        samples_per_rank = self.batch_size * (samples_per_rank // self.batch_size)

        self.index = index[rank * samples_per_rank:(rank + 1) * samples_per_rank]
        self.index_cycle = cycle(self.index)
        self.size = len(self.index)
        self.steps = self.size // self.batch_size
        print("partition:{0}, rank:{1}, sharded index size:{2}, batch_size:{3}, steps:{4}".format(partition, rank, self.size, self.batch_size, self.steps))


#        self.index = index
#        self.index_cycle = cycle(index)
#        self.size = len(index)
#        self.steps = np.ceil(self.size / batch_size)
#        # self.steps = np.ceil(self.size / batch_size / 100)

    def __len__(self):
        return self.steps

    def __getitem__(self, idx):
        shard = self.index[idx * self.batch_size:(idx + 1) * self.batch_size]
        x_list, y = self.get_slice(self.batch_size, single=self.single, partial_index=shard)
        return x_list, y

    def reset(self):
        self.index_cycle = cycle(self.index)

    def get_response(self, copy=False):
        df = self.data.df_response.iloc[self.index, :].drop(['Group'], axis=1)
        return df.copy() if copy else df

#    def get_slice(self, size=None, contiguous=True, single=False, dataframe=False):
    def get_slice(self, size=None, contiguous=True, single=False, dataframe=False, partial_index=None):
        size = size or self.size
        single = single or self.data.agg_dose
        target = self.data.agg_dose or 'Growth'

#        index = list(islice(self.index_cycle, size))
        if partial_index is not None:
            index = partial_index
        else:
            index = list(islice(self.index_cycle, size))
        df_orig = self.data.df_response.iloc[index, :]
        df = df_orig.copy()

        if not single:
            df['Swap'] = np.random.choice([True, False], df.shape[0])
            swap = df_orig['Drug2'].notnull() & df['Swap']
            df.loc[swap, 'Drug1'] = df_orig.loc[swap, 'Drug2']
            df.loc[swap, 'Drug2'] = df_orig.loc[swap, 'Drug1']
            if not self.data.agg_dose:
                df['DoseSplit'] = np.random.uniform(0.001, 0.999, df.shape[0])
                df.loc[swap, 'Dose1'] = df_orig.loc[swap, 'Dose2']
                df.loc[swap, 'Dose2'] = df_orig.loc[swap, 'Dose1']

        split = df_orig['Drug2'].isnull()
        if not single:
            df.loc[split, 'Drug2'] = df_orig.loc[split, 'Drug1']
            if not self.data.agg_dose:
                df.loc[split, 'Dose1'] = df_orig.loc[split, 'Dose1'] - np.log10(df.loc[split, 'DoseSplit'])
                df.loc[split, 'Dose2'] = df_orig.loc[split, 'Dose1'] - np.log10(1 - df.loc[split, 'DoseSplit'])

        if dataframe:
            cols = [target, 'Sample', 'Drug1', 'Drug2'] if not single else [target, 'Sample', 'Drug1']
            y = df[cols].reset_index(drop=True)
        else:
            y = values_or_dataframe(df[target], contiguous, dataframe)

        x_list = []

        if not self.data.agg_dose:
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


def find_columns_with_str(df, substr):
    col_indices = [df.columns.get_loc(col) for col in df.columns if substr in col]

    return col_indices

class FromFileDataGenerator(object):
    """Generate testing batches from loaded data
    """
    def __init__(self, df_data, indices, target_str, feature_names_list, num_features_list, batch_size=32, shuffle=True):
    
        self.batch_size = batch_size
        
        index = indices

        if shuffle:
            index = np.random.permutation(index)

        self.index = index
        self.index_cycle = cycle(index)
        self.size = len(index)
        self.steps = np.ceil(self.size / batch_size)
        
        self.num_features_list = num_features_list
        
        try : # Try to get the 'target_str' column
            target = df_data.columns.get_loc(target_str)
        except KeyError: # The 'target_str' column is not available in data file
            # No ground truth available
            y_fake = np.zeros(df_data.shape[0])
            df_data['fake_target'] = y_fake
            self.target = df_data.columns.get_loc('fake_target')
        else: # 'target_str' column is available --> use this column
            self.target = target
        
        self.df_data = df_data
        self.offset = self.compute_offset(feature_names_list)

    def compute_offset(self, feature_names):
        offset = self.df_data.shape[1]
        for name in feature_names:
            col_indices = find_columns_with_str(self.df_data, name)
            if len(col_indices) > 0:
                first_col = np.min(col_indices)
                if first_col < offset:
                    offset = first_col

        if offset == self.df_data.shape[1]:
            raise Exception('ERROR ! Feature names from model are not in file. ' \
            'These are features in model: ' + str(sorted(feature_names)) + \
            '... Exiting')

        return offset

    def reset(self):
        self.index_cycle = cycle(self.index)

    def get_response(self, copy=False):
        df = self.df_data.iloc[self.index, :]
        return df.copy() if copy else df

    def get_slice(self, size=None, contiguous=True):
    
        size = size or self.size
        index = list(islice(self.index_cycle, size))
        df_orig = self.df_data.iloc[index, :]
        df = df_orig.copy()
        
        #Features -->
        x_list = []
        start = self.offset
        # features need to be provided in the partitions expected by the model
        for i,numf in enumerate(self.num_features_list):
            end = start + numf
            mat = df.iloc[:,start:end].values
            if contiguous:
                mat = np.ascontiguousarray(mat)
            x_list.append(mat)
            start = end

        # Target
        mat = df.iloc[:,self.target].values
        if contiguous:
            mat = np.ascontiguousarray(mat)
        y = mat
    
        # print(x_list, y)
        return x_list, y


    def flow(self, single=False):
        while 1:
            x_list, y = self.get_slice(self.batch_size)
            yield x_list, y

