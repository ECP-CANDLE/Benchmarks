"""
    File Name:          UnoPytorch/basic_dataset.py
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               8/16/18
    Python Version:     3.6.6
    File Description:

"""

import numpy as np
import pandas as pd
import torch.utils.data as data


class DataFrameDataset(data.Dataset):
    """This class implements a basic PyTorch dataset from given dataframe.

    Note that this class does not take care of any form of data processing.
    It merely stores data from given dataframe in the form of np.array in
    certain data type.

    Also note that the given dataframe should be purely numeric.
    """

    def __init__(
            self,
            dataframe: pd.DataFrame,
            ram_dtype: type = np.float16,
            out_dtype: type = np.float32, ):
        """dataset = DataFrameDataset(dataframe)

        This function initializes a dataset from given dataframe. Upon init,
        the function will convert dataframe into numpy array for faster
        data retrieval. To save the ram space and make data fetching even
        faster, it will store the data in certain dtype to save some
        unnecessary precision bits.

        However, it will still convert the data into out_dtype during slicing.

        Args:
            dataframe (pd.DataFrame): dataframe for the dataset.
            ram_dtype (type): dtype for data storage in RAM.
            out_dtype (type): dtype for data output during slicing.
        """

        self.__data = dataframe.values.astype(ram_dtype)
        self.__out_dtype = out_dtype
        self.__len = len(self.__data)

    def __len__(self):
        return self.__len

    def __getitem__(self, index):
        return self.__data[index].astype(self.__out_dtype)
