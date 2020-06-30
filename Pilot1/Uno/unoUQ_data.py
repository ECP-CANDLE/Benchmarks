
import numpy as np
#import pandas as pd

from itertools import cycle, islice


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
