'''
This module contains functions to enable in memory sharding
of data. The basic idea is to use the PMI_RANK to shift a
starting position, and from there save every Nth sample where
N is the number of nodes being used.

It is assumed that node level parallism on however many GPUs
a node has is handled by the framework (ie tensorflow).

Example:
shard = slice_total_gpus(shift_to_rank(d,_rank),_tgpus))
'''

import numpy as np

def shift_to_rank (d, _PMI_RANK=0):
    
    '''
    In a dataframe shape (10,2), pmi rank of 0 will not pop
    any off the dataframe. A pmi rank of 9 will leave the
    last sample in the dataframe. A pmi rank of 10 will pop
    all the samples off the dataframe.
    '''
    print ('in shift_to_rank, _PMI_RANK = {}'.format(_PMI_RANK))
    print ('in shift_to_rank, d.shape = {}'.format(d.shape))
    strd = d[_PMI_RANK:,:]
    return strd

def slice_total_gpus(d, _TOT_NODES=1):

    '''
    In a dataframe shape (10,2) and total nodes equal to 4, every
    4th element in the dataframe is saved. The first element
    is always saved, then skip total gpus. If total gpus is 1,
    all elements are kept.
    '''
    
    return d[::_TOT_NODES]


if __name__ == "__main__":

    # This is a test of the last rank
    _rank = 0 # (0,1,2,3)
    _tgpus = 4
    d = np.array(np.random.rand(10,2))
    
    print(d.shape)
    print(d)
    print('shift to rank {}'.format(_rank))
    print(shift_to_rank(d,_rank))
    print('slice to total gpus {}'.format(_tgpus))
    print(slice_total_gpus(d,_tgpus))
    print('shift to rank {} and slice to gotal gpus {}'.format(_rank, _tgpus))
    print(slice_total_gpus(shift_to_rank(d,_rank),_tgpus))
