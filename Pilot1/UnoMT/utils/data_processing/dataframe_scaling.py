"""
    File Name:          UnoPytorch/dataframe_scaling.py
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               8/13/18
    Python Version:     3.6.6
    File Description:

"""

import logging
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler


logger = logging.getLogger(__name__)
SCALING_METHODS = ['none', 'std', 'minmax']


def scale_dataframe(dataframe: pd.DataFrame,
                    scaling_method: str,):
    """new_df = dataframe_scaling(old_df, 'std')

    Scaling features in dataframe according to specific scaling strategy.
    TODO: More scaling options and selective feature(col) scaling for dataframe.

    Args:
        dataframe (pandas.Dataframe): dataframe to be scaled.
        scaling_method (str): 'std', 'minmax', etc.

    Returns:
        pandas.Dataframe: scaled dataframe.
    """
    scaling_method = scaling_method.lower()

    if scaling_method.lower() == 'none':
        return dataframe
    elif scaling_method.lower() == 'std':
        scaler = StandardScaler()
    elif scaling_method.lower() == 'minmax':
        scaler = MinMaxScaler()
    else:
        logger.error('Unknown scaling method %s' % scaling_method,
                     exc_info=True)
        return dataframe

    if len(dataframe.shape) == 1:
        dataframe = scaler.fit_transform(dataframe.values.reshape(-1, 1))
    else:
        dataframe[:] = scaler.fit_transform(dataframe[:])

    return dataframe
