from __future__ import absolute_import

import numpy as np


def generate_index_distribution(numTrain, numTest, numValidation, params):
    """ Generates a vector of indices to partition the data for training.
        NO CHECKING IS DONE: it is assumed that the data could be partitioned
        in the specified blocks and that the block indices describe a coherent
        partition.
        
        Parameters
        ----------
        numTrain : int
            Number of training data points
        numTest : int
            Number of testing data points
        numValidation : int
            Number of validation data points (may be zero)
        params : dictionary with parameters
            Contains the keywords that control the behavior of the function
            (uq_train_fr, uq_valid_fr, uq_test_fr for fraction specification,
            uq_train_vec, uq_valid_vec, uq_test_vec for block list specification, and
            uq_train_bks, uq_valid_bks, uq_test_bks for block number specification)
            
        Return
        ----------
        indexTrain : int numpy array
            Indices for data in training
        indexValidation : int numpy array
            Indices for data in validation (if any)
        indexTest : int numpy array
            Indices for data in testing (if merging)
    """
    if all (k in params for k in ('uq_train_fr', 'uq_valid_fr', 'uq_test_fr')):
        # specification by fraction
        print("Computing UQ cross-validation - Distributing by FRACTION")
        return generate_index_distribution_from_fraction(numTrain, numTest, numValidation, params)
    elif all (k in params for k in ('uq_train_vec', 'uq_valid_vec', 'uq_test_vec')):
        # specification by block list
        print("Computing UQ cross-validation - Distributing by BLOCK LIST")
        return generate_index_distribution_from_block_list(numTrain, numTest, numValidation, params)
    elif all (k in params for k in ('uq_train_bks', 'uq_valid_bks', 'uq_test_bks')):
        # specification by block size
        print("Computing UQ cross-validation - Distributing by BLOCK NUMBER")
        return generate_index_distribution_from_blocks(numTrain, numTest, numValidation, params)
    else:
        print("ERROR !! No consistent UQ parameter specification found !! ... exiting ")
        raise KeyError("No valid triplet of ('uq_train_*', 'uq_valid_*', 'uq_test_*') found. (* is any of fr, vec or bks)")


def generate_index_distribution_from_fraction(numTrain, numTest, numValidation, params):
    """ Generates a vector of indices to partition the data for training.
        It checks that the fractions provided are (0, 1) and add up to 1.
       
        Parameters
        ----------
        numTrain : int
            Number of training data points
        numTest : int
            Number of testing data points
        numValidation : int
            Number of validation data points (may be zero)
        params : dictionary with parameters
            Contains the keywords that control the behavior of the function
            (uq_train_fr, uq_valid_fr, uq_test_fr)
            
        Return
        ----------
        indexTrain : int numpy array
            Indices for data in training
        indexValidation : int numpy array
            Indices for data in validation (if any)
        indexTest : int numpy array
            Indices for data in testing (if merging)
    """

    # Extract required parameters
    fractionTrain = params['uq_train_fr']
    fractionValidation = params['uq_valid_fr']
    fractionTest = params['uq_test_fr']
    
    if (fractionTrain < 0.) or (fractionTrain > 1.):
        raise ValueError('uq_train_fr is not in (0, 1) range. uq_train_fr: ', fractionTrain)
    if (fractionValidation < 0.) or (fractionValidation > 1.):
        raise ValueError('uq_valid_fr is not in (0, 1) range. uq_valid_fr: ', fractionValidation)
    if (fractionTest < 0.) or (fractionTest > 1.):
        raise ValueError('uq_test_fr is not in (0, 1) range. uq_test_fr: ', fractionTest)

    fractionSum = fractionTrain + fractionValidation + fractionTest
    if (fractionSum > 1.) or (fractionSum < 1.):
        raise ValueError('Specified UQ fractions (uq_train_fr, uq_valid_fr, uq_test_fr) do not add up to 1. No cross-validation partition is computed ! sum:', fractionSum)

    # Determine data size and block size
    if fractionTest > 0:
        # Use all data and re-distribute the partitions
        numData = numTrain + numValidation + numTest
    else:
        # Preserve test partition
        numData = numTrain + numValidation
    
    sizeTraining = int(np.round(numData * fractionTrain))
    sizeValidation = int(np.round(numData * fractionValidation))

    # Fill partition indices
    # Fill train partition    
    Folds = np.arange(numData)
    np.random.shuffle(Folds)
    indexTrain = Folds[:sizeTraining]
    # Fill validation partition
    indexValidation = None
    if fractionValidation > 0:
        indexValidation = Folds[sizeTraining:sizeTraining+sizeValidation]
    # Fill test partition
    indexTest = None
    if fractionTest > 0:
        indexTest = Folds[sizeTraining+sizeValidation:]
    
    return indexTrain, indexValidation, indexTest


def generate_index_distribution_from_blocks(numTrain, numTest, numValidation, params):
    """ Generates a vector of indices to partition the data for training.
        NO CHECKING IS DONE: it is assumed that the data could be partitioned
        in the specified block quantities and that the block quantities describe a
        coherent partition.
        
        Parameters
        ----------
        numTrain : int
            Number of training data points
        numTest : int
            Number of testing data points
        numValidation : int
            Number of validation data points (may be zero)
        params : dictionary with parameters
            Contains the keywords that control the behavior of the function
            (uq_train_bks, uq_valid_bks, uq_test_bks)
            
        Return
        ----------
        indexTrain : int numpy array
            Indices for data in training
        indexValidation : int numpy array
            Indices for data in validation (if any)
        indexTest : int numpy array
            Indices for data in testing (if merging)
    """

    # Extract required parameters
    numBlocksTrain = params['uq_train_bks']
    numBlocksValidation = params['uq_valid_bks']
    numBlocksTest = params['uq_test_bks']
    numBlocksTotal = numBlocksTrain + numBlocksValidation + numBlocksTest

    # Determine data size and block size
    if numBlocksTest > 0:
        # Use all data and re-distribute the partitions
        numData = numTrain + numValidation + numTest
    else:
        # Preserve test partition
        numData = numTrain + numValidation
    
    blockSize = (numData + numBlocksTotal // 2) // numBlocksTotal # integer division with rounding
    remainder = numData - blockSize * numBlocksTotal
    if remainder != 0:
        print("Warning ! Requested partition does not distribute data evenly between blocks. "
              "Testing (if specified) or Validation (if specified) will use different block size.")

    sizeTraining = numBlocksTrain * blockSize
    sizeValidation = numBlocksValidation * blockSize

    # Fill partition indices
    # Fill train partition    
    Folds = np.arange(numData)
    np.random.shuffle(Folds)
    indexTrain = Folds[:sizeTraining]
    # Fill validation partition
    indexValidation = None
    if numBlocksValidation > 0:
        indexValidation = Folds[sizeTraining:sizeTraining+sizeValidation]
    # Fill test partition
    indexTest = None
    if numBlocksTest > 0:
        indexTest = Folds[sizeTraining+sizeValidation:]
    
    return indexTrain, indexValidation, indexTest



def generate_index_distribution_from_block_list(numTrain, numTest, numValidation, params):
    """ Generates a vector of indices to partition the data for training.
        NO CHECKING IS DONE: it is assumed that the data could be partitioned
        in the specified list of blocks and that the block indices describe a
        coherent partition.
        
        Parameters
        ----------
        numTrain : int
            Number of training data points
        numTest : int
            Number of testing data points
        numValidation : int
            Number of validation data points (may be zero)
        params : dictionary with parameters
            Contains the keywords that control the behavior of the function
            (uq_train_vec, uq_valid_vec, uq_test_vec)
            
        Return
        ----------
        indexTrain : int numpy array
            Indices for data in training
        indexValidation : int numpy array
            Indices for data in validation (if any)
        indexTest : int numpy array
            Indices for data in testing (if merging)
    """

    # Extract required parameters
    blocksTrain = params['uq_train_vec']
    blocksValidation = params['uq_valid_vec']
    blocksTest = params['uq_test_vec']
    
    # Determine data size and block size
    numBlocksTrain = len(blocksTrain)
    numBlocksValidation = len(blocksValidation)
    numBlocksTest = len(blocksTest)
    numBlocksTotal = numBlocksTrain + numBlocksValidation + numBlocksTest

    if numBlocksTest > 0:
        # Use all data and re-distribute the partitions
        numData = numTrain + numValidation + numTest
    else:
        # Preserve test partition
        numData = numTrain + numValidation
    
    blockSize = (numData + numBlocksTotal // 2) // numBlocksTotal # integer division with rounding
    remainder = numData - blockSize * numBlocksTotal
    if remainder != 0:
        print("Warning ! Requested partition does not distribute data evenly between blocks. "
              "Last block will have different size.")
    if remainder < 0:
        remainder = 0

    # Fill partition indices
    # Fill train partition
    maxSizeTrain = blockSize * numBlocksTrain + remainder
    indexTrain = fill_array(blocksTrain, maxSizeTrain, numData, numBlocksTotal, blockSize)
    # Fill validation partition
    indexValidation = None
    if numBlocksValidation > 0:
        maxSizeValidation = blockSize * numBlocksValidation  + remainder
        indexValidation = fill_array(blocksValidation, maxSizeValidation, numData, numBlocksTotal, blockSize)
    # Fill test partition
    indexTest = None
    if numBlocksTest > 0:
        maxSizeTest = blockSize * numBlocksTest + remainder
        indexTest = fill_array(blocksTest, maxSizeTest, numData, numBlocksTotal, blockSize)

    return indexTrain, indexValidation, indexTest



def compute_limits(numdata, numblocks, blocksize, blockn):
    """ Generates the limit of indices corresponding to a
        specific block. It takes into account the non-exact
        divisibility of numdata into numblocks letting the
        last block to take the extra chunk.
        
        Parameters
        ----------
        numdata : int
            Total number of data points to distribute
        numblocks : int
            Total number of blocks to distribute into
        blocksize : int
            Size of data per block
        blockn : int
            Index of block, from 0 to numblocks-1
            
        Return
        ----------
        start : int
            Position to start assigning indices
        end : int
            One beyond position to stop assigning indices
    """
    start = blockn * blocksize
    end = start + blocksize
    if blockn == (numblocks-1): # last block gets the extra
        end = numdata

    return start, end


def fill_array(blocklist, maxsize, numdata, numblocks, blocksize):
    """ Fills a new array of integers with the indices corresponding
        to the specified block structure.
        
        Parameters
        ----------
        blocklist : list
            List of integers describen the block indices that
            go into the array
        maxsize : int
            Maximum possible length for the partition (the size of the
            common block size plus the remainder, if any).
        numdata : int
            Total number of data points to distribute
        numblocks : int
            Total number of blocks to distribute into
        blocksize : int
            Size of data per block
            
        Return
        ----------
        indexArray : int numpy array
            Indices for specific data partition. Resizes the array
            to the correct length.
    """

    indexArray = np.zeros(maxsize, np.int)

    offset = 0
    for i in blocklist:
        start, end = compute_limits(numdata, numblocks, blocksize, i)
        length = end - start
        indexArray[offset:offset+length] = np.arange(start, end)
        offset += length

    return indexArray[:offset]









