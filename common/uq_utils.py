from __future__ import absolute_import

import numpy as np
from scipy.stats import pearsonr, spearmanr
from scipy import signal
from scipy.interpolate import InterpolatedUnivariateSpline

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

    tol = 1e-7

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
    #if (fractionSum > 1.) or (fractionSum < 1.):
    if abs(fractionSum-1.) > tol:
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


###### UTILS for COMPUTATION OF EMPIRICAL CALIBRATION

def compute_statistics_homoscedastic(df_data,
                                     col_true=0,
                                     col_pred=6,
                                     col_std_pred=7,
                                     ):
    """ Extracts ground truth, mean predition, error and
        standard deviation of prediction from inference
        data frame. The latter includes the statistics
        over all the inference realizations.
        
        Parameters
        ----------
        df_data : pandas data frame
            Data frame generated by current CANDLE inference
            experiments. Indices are hard coded to agree with
            current CANDLE version. (The inference file usually
            has the name: <model>_pred.tsv).
        col_true : integer
            Index of the column in the data frame where the true 
            value is stored (Default: 0, index in current CANDLE format).
        col_pred : integer
            Index of the column in the data frame where the predicted 
            value is stored (Default: 6, index in current CANDLE format).
        col_std_pred : integer
            Index of the column in the data frame where the standard 
            deviation of the predicted values is stored (Default: 7,
            index in current CANDLE format).
            
        Return
        ----------
        Ytrue : numpy array
            Array with true (observed) values
        Ypred : numpy array
            Array with predicted values.
        yerror : numpy array
            Array with errors computed (observed - predicted).
        sigma : numpy array
            Array with standard deviations learned with deep learning
            model. For homoscedastic inference this corresponds to the
            std value computed from prediction (and is equal to the 
            following returned variable).
        Ypred_std : numpy array
            Array with standard deviations computed from regular
            (homoscedastic) inference.
        pred_name : string
            Name of data colum or quantity predicted (as extracted
            from the data frame using the col_true index).
    """

    Ytrue = df_data.iloc[:,col_true].values
    print('Ytrue shape: ', Ytrue.shape)
    pred_name = df_data.columns[col_true]
    Ypred = df_data.iloc[:,col_pred].values
    print('Ypred shape: ', Ypred.shape)
    Ypred_std = df_data.iloc[:,col_std_pred].values
    print('Ypred_std shape: ', Ypred_std.shape)
    yerror = Ytrue - Ypred
    print('yerror shape: ', yerror.shape)
    sigma = Ypred_std # std
    MSE = np.mean((Ytrue - Ypred)**2)
    print('MSE: ', MSE)
    MSE_STD = np.std((Ytrue - Ypred)**2)
    print('MSE_STD: ', MSE_STD)
    # p-value 'not entirely reliable, reasonable for datasets > 500'
    spearman_cc, pval = spearmanr(Ytrue, Ypred)
    print('Spearman CC: %f, p-value: %e' % (spearman_cc, pval))

    return Ytrue, Ypred, yerror, sigma, Ypred_std, pred_name


def compute_statistics_homoscedastic_all(df_data,
                                     col_true=4,
                                     col_pred_start=6
                                     ):
    """ Extracts ground truth, mean predition, error and
        standard deviation of prediction from inference
        data frame. The latter includes all the individual
        inference realizations.
        
        Parameters
        ----------
        df_data : pandas data frame
            Data frame generated by current CANDLE inference
            experiments. Indices are hard coded to agree with
            current CANDLE version. (The inference file usually
            has the name: <model>.predicted_INFER.tsv).
        col_true : integer
            Index of the column in the data frame where the true
            value is stored (Default: 4, index in current HOM format).
        col_pred_start : integer
            Index of the column in the data frame where the first predicted
            value is stored. All the predicted values during inference
            are stored (Default: 6 index, in current HOM format).
            
        Return
        ----------
        Ytrue : numpy array
            Array with true (observed) values
        Ypred : numpy array
            Array with predicted values.
        yerror : numpy array
            Array with errors computed (observed - predicted).
        sigma : numpy array
            Array with standard deviations learned with deep learning
            model. For homoscedastic inference this corresponds to the
            std value computed from prediction (and is equal to the
            following returned variable).
        Ypred_std : numpy array
            Array with standard deviations computed from regular
            (homoscedastic) inference.
        pred_name : string
            Name of data colum or quantity predicted (as extracted
            from the data frame using the col_true index).
    """

    Ytrue = df_data.iloc[:,col_true].values
    print('Ytrue shape: ', Ytrue.shape)
    pred_name = df_data.columns[col_true]
    Ypred_mean_ = np.mean(df_data.iloc[:,col_pred_start:], axis=1)
    Ypred_mean = Ypred_mean_.values
    print('Ypred_mean shape: ', Ypred_mean.shape)
    Ypred_std_ = np.std(df_data.iloc[:,col_pred_start:], axis=1)
    Ypred_std = Ypred_std_.values
    print('Ypred_std shape: ', Ypred_std.shape)
    yerror = Ytrue - Ypred_mean
    print('yerror shape: ', yerror.shape)
    sigma = Ypred_std # std
    MSE = np.mean((Ytrue - Ypred_mean)**2)
    print('MSE: ', MSE)
    MSE_STD = np.std((Ytrue - Ypred_mean)**2)
    print('MSE_STD: ', MSE_STD)
    # p-value 'not entirely reliable, reasonable for datasets > 500'
    spearman_cc, pval = spearmanr(Ytrue, Ypred_mean)
    print('Spearman CC: %f, p-value: %e' % (spearman_cc, pval))

    return Ytrue, Ypred_mean, yerror, sigma, Ypred_std, pred_name


def compute_statistics_heteroscedastic(df_data,
                                     col_true=4,
                                     col_pred_start=6,
                                     col_std_pred_start=7,
                                     ):
    """ Extracts ground truth, mean predition, error, standard
        deviation of prediction and predicted (learned) standard
        deviation from inference data frame. The latter includes
        all the individual inference realizations.
        
        Parameters
        ----------
        df_data : pandas data frame
            Data frame generated by current heteroscedastic inference
            experiments. Indices are hard coded to agree with
            current version. (The inference file usually
            has the name: <model>.predicted_INFER_HET.tsv).
        col_true : integer
            Index of the column in the data frame where the true
            value is stored (Default: 4, index in current HET format).
        col_pred_start : integer
            Index of the column in the data frame where the first predicted
            value is stored. All the predicted values during inference
            are stored and are interspaced with standard deviation
            predictions (Default: 6 index, step 2, in current HET format).
        col_std_pred_start : integer
            Index of the column in the data frame where the first predicted
            standard deviation value is stored. All the predicted values
            during inference are stored and are interspaced with predictions
            (Default: 7 index, step 2, in current HET format).
            
        Return
        ----------
        Ytrue : numpy array
            Array with true (observed) values
        Ypred : numpy array
            Array with predicted values.
        yerror : numpy array
            Array with errors computed (observed - predicted).
        sigma : numpy array
            Array with standard deviations learned with deep learning
            model. For homoscedastic inference this corresponds to the
            std value computed from prediction (and is equal to the
            following returned variable).
        Ypred_std : numpy array
            Array with standard deviations computed from regular
            (homoscedastic) inference.
        pred_name : string
            Name of data colum or quantity predicted (as extracted
            from the data frame using the col_true index).
    """

    Ytrue = df_data.iloc[:,col_true].values
    print('Ytrue shape: ', Ytrue.shape)
    pred_name = df_data.columns[col_true]
    Ypred_mean_ = np.mean(df_data.iloc[:,col_pred_start::2], axis=1)
    Ypred_mean = Ypred_mean_.values
    print('Ypred shape: ', Ypred_mean.shape)
    Ypred_std_ = np.std(df_data.iloc[:,col_pred_start::2], axis=1)
    Ypred_std = Ypred_std_.values
    print('Ypred_std shape: ', Ypred_std.shape)
    yerror = Ytrue - Ypred_mean
    print('yerror shape: ', yerror.shape)
    s_ = df_data.iloc[:,col_std_pred_start::2]
    s_mean = np.mean(s_, axis=1)
    var = np.exp(s_mean.values) # variance
    sigma = np.sqrt(var) # std
    print('sigma shape: ', sigma.shape)
    MSE = np.mean((Ytrue - Ypred_mean)**2)
    print('MSE: ', MSE)
    MSE_STD = np.std((Ytrue - Ypred_mean)**2)
    print('MSE_STD: ', MSE_STD)
    # p-value 'not entirely reliable, reasonable for datasets > 500'
    spearman_cc, pval = spearmanr(Ytrue, Ypred_mean)
    print('Spearman CC: %f, p-value: %e' % (spearman_cc, pval))

    return Ytrue, Ypred_mean, yerror, sigma, Ypred_std, pred_name


def compute_statistics_quantile(df_data,
                                     sigma_divisor=2.56,
                                     col_true=4,
                                     col_pred_start=6
                                     ):
    """ Extracts ground truth, 50th percentile mean predition,
        low percentile and high percentile mean prediction
        (usually 10th percentile and 90th percentile respectively),
        error (using 50th percentile), standard deviation of
        prediction (using 50th percentile) and predicted (learned)
        standard deviation from interdecile range in inference data frame.
        The latter includes all the individual inference realizations.
        
        Parameters
        ----------
        df_data : pandas data frame
            Data frame generated by current quantile inference
            experiments. Indices are hard coded to agree with
            current version. (The inference file usually
            has the name: <model>.predicted_INFER_QTL.tsv).
        sigma_divisor : float
            Divisor to convert from the intercedile range to the corresponding
            standard deviation for a Gaussian distribution.
            (Default: 2.56, consisten with an interdecile range computed from
            the difference between the 90th and 10th percentiles).
        col_true : integer
            Index of the column in the data frame where the true
            value is stored (Default: 4, index in current QTL format).
        col_pred_start : integer
            Index of the column in the data frame where the first predicted
            value is stored. All the predicted values during inference
            are stored and are interspaced with other percentile
            predictions (Default: 6 index, step 3, in current QTL format).
            
        Return
        ----------
        Ytrue : numpy array
            Array with true (observed) values
        Ypred : numpy array
            Array with predicted values (based on the 50th percentile).
        yerror : numpy array
            Array with errors computed (observed - predicted).
        sigma : numpy array
            Array with standard deviations learned with deep learning
            model. This corresponds to the interdecile range divided
            by the sigma divisor.
        Ypred_std : numpy array
            Array with standard deviations computed from regular
            (homoscedastic) inference.
        pred_name : string
            Name of data colum or quantity predicted (as extracted
            from the data frame using the col_true index).
        Ypred_Lp_mean : numpy array
            Array with predicted values of the lower percentile
            (usually the 10th percentile).
        Ypred_Hp_mean : numpy array
            Array with predicted values of the higher percentile
            (usually the 90th percentile).
    """

    Ytrue = df_data.iloc[:,col_true].values
    print('Ytrue shape: ', Ytrue.shape)
    pred_name = df_data.columns[col_true]
    Ypred_50q_mean = np.mean(df_data.iloc[:,col_pred_start::3], axis=1)
    Ypred_mean = Ypred_50q_mean.values
    print('Ypred shape: ', Ypred_mean.shape)
    Ypred_Lp_mean_ = np.mean(df_data.iloc[:,col_pred_start+1::3], axis=1)
    Ypred_Hp_mean_ = np.mean(df_data.iloc[:,col_pred_start+2::3], axis=1)
    Ypred_Lp_mean = Ypred_Lp_mean_.values
    Ypred_Hp_mean = Ypred_Hp_mean_.values
    interdecile_range = Ypred_Hp_mean - Ypred_Lp_mean
    sigma = interdecile_range / sigma_divisor
    print('sigma shape: ', sigma.shape)
    yerror = Ytrue - Ypred_mean
    print('yerror shape: ', yerror.shape)
    Ypred_std_ = np.std(df_data.iloc[:,col_pred_start::3], axis=1)
    Ypred_std = Ypred_std_.values
    print('Ypred_std shape: ', Ypred_std.shape)
    MSE = np.mean((Ytrue - Ypred_mean)**2)
    print('MSE: ', MSE)
    MSE_STD = np.std((Ytrue - Ypred_mean)**2)
    print('MSE_STD: ', MSE_STD)
    # p-value 'not entirely reliable, reasonable for datasets > 500'
    spearman_cc, pval = spearmanr(Ytrue, Ypred_mean)
    print('Spearman CC: %f, p-value: %e' % (spearman_cc, pval))

    return Ytrue, Ypred_mean, yerror, sigma, Ypred_std, pred_name, Ypred_Lp_mean, Ypred_Hp_mean


def split_data_for_empirical_calibration(Ytrue, Ypred, sigma, cal_split=0.8):
    """ Extracts a portion of the arrays provided for the computation
        of the calibration and reserves the remainder portion
        for testing.
        
        Parameters
        ----------
        Ytrue : numpy array
            Array with true (observed) values
        Ypred : numpy array
            Array with predicted values.
        sigma : numpy array
            Array with standard deviations learned with deep learning
            model (or std value computed from prediction if homoscedastic
            inference).
        cal_split : float
             Split of data to use for estimating the calibration relationship.
             It is assumet that it will be a value in (0, 1).
             (Default: use 80% of predictions to generate empirical
             calibration).
            
        Return
        ----------
        index_perm_total : numpy array
            Random permutation of the array indices. The first 'num_cal'
            of the indices correspond to the samples that are used for
            calibration, while the remainder are the samples reserved 
            for calibration testing.
        pSigma_cal : numpy array
            Part of the input sigma array to use for calibration.
        pSigma_test : numpy array
            Part of the input sigma array to reserve for testing.
        pPred_cal : numpy array
            Part of the input Ypred array to use for calibration.
        pPred_test : numpy array
            Part of the input Ypred array to reserve for testing.
        true_cal : numpy array
            Part of the input Ytrue array to use for calibration.
        true_test : numpy array
            Part of the input Ytrue array to reserve for testing.
    """

    # shuffle data for calibration
    num_pred_total = sigma.shape[0]
    num_cal = np.int(num_pred_total * cal_split)
    index_perm_total = np.random.permutation(range(num_pred_total))

    # Permute data
    pSigma_perm_all = sigma[index_perm_total]
    pPred_perm_all = Ypred[index_perm_total]
    true_perm_all = Ytrue[index_perm_total]

    # Split in calibration and testing
    pSigma_cal = pSigma_perm_all[:num_cal]
    pSigma_test = pSigma_perm_all[num_cal:]
    pPred_cal = pPred_perm_all[:num_cal]
    pPred_test = pPred_perm_all[num_cal:]
    true_cal = true_perm_all[:num_cal]
    true_test = true_perm_all[num_cal:]

    print('Size of calibration set: ', true_cal.shape)
    print('Size of test set: ', true_test.shape)

    return index_perm_total, pSigma_cal, pSigma_test, pPred_cal, pPred_test, true_cal, true_test


def compute_empirical_calibration(pSigma_cal, pPred_cal, true_cal, bins, coverage_percentile):
    """ Use the arrays provided to estimate an empirical mapping
        between standard deviation and absolute value of error,
        both of which have been observed during inference. Since
        most of the times the raw statistics per bin are very noisy,
        a smoothing step (based on scipy's savgol filter) is performed.
        
        Parameters
        ----------
        pSigma_cal : numpy array
            Part of the standard deviations array to use for calibration.
        pPred_cal : numpy array
            Part of the predictions array to use for calibration.
        true_cal : numpy array
            Part of the true (observed) values array to use for calibration.
        bins : int
            Number of bins to split the range of standard deviations
            included in pSigma_cal array.
        coverage_percentile : float
            Value to use for estimating coverage when evaluating the percentiles 
            of the observed absolute value of errors.
            
        Return
        ----------
        mean_sigma : numpy array
            Array with the mean standard deviations computed per bin.
        min_sigma : numpy array
            Array with the minimum standard deviations computed per bin.
        max_sigma : numpy array
            Array with the maximum standard deviations computed per bin.
        error_thresholds : numpy array
            Thresholds of the errors computed to attain a certain
            error coverage per bin.
        err_err : numpy array
            Error bars in errors (one standard deviation for a binomial
            distribution estimated by bin vs. the other bins) for the 
            calibration error.
        error_thresholds_smooth : numpy array
            Thresholds of the errors computed to attain a certain
            error coverage per bin after a smoothed operation is applied
            to the frequently noisy bin-based estimations.
        sigma_start_index : non-negative integer
            Index in the mean_sigma array that defines the start of
            the valid empirical calibration interval (i.e. index to
            the smallest std for which a meaningful error mapping
            is obtained).
        sigma_end_index : non-negative integer
            Index in the mean_sigma array that defines the end of
            the valid empirical calibration interval (i.e. index to
            the largest std for which a meaningful error mappping 
            is obtained).
        s_interpolate : scipy.interpolate python object
            A python object from scipy.interpolate that computes a
            univariate spline (InterpolatedUnivariateSpline) constructed
            to express the mapping from standard deviation to error. This
            spline is generated during the computational empirical
            calibration procedure.
    """

    index_sigma_cal = np.argsort(pSigma_cal)
    pSigma_cal_ordered_ = pSigma_cal[index_sigma_cal]
    Er_vect_cal_ = np.abs(true_cal - pPred_cal)
    Er_vect_cal_orderedSigma_ = Er_vect_cal_[index_sigma_cal]

    minL_sigma = np.min(pSigma_cal_ordered_)
    maxL_sigma = np.max(pSigma_cal_ordered_)
    print('Complete Sigma range --> Min: %f, Max: %f' % (minL_sigma, maxL_sigma))

    # Bin statistics for error and sigma
    mean_sigma, min_sigma, max_sigma, error_thresholds, err_err = bining_for_calibration(pSigma_cal_ordered_,
                                minL_sigma,
                                maxL_sigma,
                                Er_vect_cal_orderedSigma_,
                                bins,
                                coverage_percentile)

    # smooth error function
    #scipy.signal.savgol_filter(x, window_length, polyorder,
    #deriv=0, delta=1.0, axis=-1, mode='interp', cval=0.0)
    #error_thresholds_smooth = signal.savgol_filter(error_thresholds, 5, 1)
    error_thresholds_smooth = signal.savgol_filter(error_thresholds, 5, 1, mode='nearest')

    # Build Interpolant over smooth plot (this will become the calibration function)
    s_interpolate = InterpolatedUnivariateSpline(mean_sigma, error_thresholds_smooth)
    # Determine limits of calibration (i.e. monotonicity range)
    sigma_start_index, sigma_end_index = computation_of_valid_calibration_interval(error_thresholds, error_thresholds_smooth, err_err)
    
    print('Range of valid sigma: %.6f --> %.6f' % (mean_sigma[sigma_start_index], mean_sigma[sigma_end_index]))

    return mean_sigma, min_sigma, max_sigma, error_thresholds, err_err, error_thresholds_smooth, sigma_start_index, sigma_end_index, s_interpolate



def bining_for_calibration(pSigma_cal_ordered_, minL_sigma,
                            maxL_sigma, Er_vect_cal_orderedSigma_,
                            bins, coverage_percentile):
    """ Bin the values of the standard deviations observed during
        inference and estimate a specified coverage percentile
        in the absolute error (observed during inference as well).
        Bins that have less than 50 samples are merged until they 
        surpass this threshold.
        
        Parameters
        ----------
        pSigma_cal_ordered_ : numpy array
            Array of standard deviations ordered in ascending way.
        minL_sigma : float
            Minimum value of standard deviations included in 
            pSigma_cal_ordered_ array.
        maxL_sigma : numpy array
            Maximum value of standard deviations included in
            pSigma_cal_ordered_ array.
        Er_vect_cal_orderedSigma_ : numpy array
            Array ob absolute value of errors corresponding with
            the array of ordered standard deviations.
        bins : int
            Number of bins to split the range of standard deviations
            included in pSigma_cal_ordered_ array.
        coverage_percentile : float
            Value to use for estimating coverage when evaluating the percentiles 
            of the observed absolute value of errors.
            
        Return
        ----------
        mean_sigma : numpy array
            Array with the mean standard deviations computed per bin.
        min_sigma : numpy array
            Array with the minimum standard deviations computed per bin.
        max_sigma : numpy array
            Array with the maximum standard deviations computed per bin.
        error_thresholds : numpy array
            Thresholds of the errors computed to attain a certain
            error coverage per bin.
        err_err : numpy array
            Error bars in errors (one standard deviation for a binomial
            distribution estimated by bin vs. the other bins) for the 
            calibration error.
    """

    #thresholds = np.logspace(np.log10(minL_sigma), np.log10(maxL_sigma), num=bins)
    thresholds = np.linspace(minL_sigma, maxL_sigma, num=bins)
    classes = np.digitize(pSigma_cal_ordered_, thresholds)
    Nbin = np.zeros(bins+1)
    for i in range(bins+1):
        indices = (classes == i)
        Nbin[i] = indices.sum()

    # Repair bins
    new_thresholds_l = []
    new_nbins_l = []
    sumN = 0
    for i in range(Nbin.shape[0]):
        sumN += Nbin[i]
        if sumN > 50:
            if i > (thresholds.shape[0] - 1):
                new_thresholds_l.append(thresholds[-1])
            else:
                new_thresholds_l.append(thresholds[i])
            new_nbins_l.append(sumN)
            sumN = 0
    new_thresholds = np.array(new_thresholds_l)
    new_nbins = np.array(new_nbins_l)
    new_thresholds[-1] = thresholds[-1]
    new_nbins[-1] += sumN

    #
    classes = np.digitize(pSigma_cal_ordered_, new_thresholds[:-1])
    error_thresholds = -1. * np.ones(new_nbins.shape[0])
    mean_sigma = -1. * np.ones(new_nbins.shape[0])
    min_sigma = -1. * np.ones(new_nbins.shape[0])
    max_sigma = -1. * np.ones(new_nbins.shape[0])
    err_err = -1. * np.ones(new_nbins.shape[0])
    Ncal = pSigma_cal_ordered_.shape[0]
    for i in range(error_thresholds.shape[0]):
        indices = (classes == i)
        n_aux = indices.sum()
        assert n_aux == new_nbins[i]
        print('Points in bin %d: %d' % (i, n_aux))
        mean_sigma[i] = np.mean(pSigma_cal_ordered_[indices])
        min_sigma[i] = np.min(pSigma_cal_ordered_[indices])
        max_sigma[i] = np.max(pSigma_cal_ordered_[indices])
        error_thresholds[i] = np.percentile(Er_vect_cal_orderedSigma_[indices], coverage_percentile)
        err_err[i] = np.sqrt(new_nbins[i] * (Ncal - new_nbins[i])) / Ncal * error_thresholds[i]

    return mean_sigma, min_sigma, max_sigma, error_thresholds, err_err


def computation_of_valid_calibration_interval(error_thresholds, error_thresholds_smooth, err_err):
    """ Function that estimates the empirical range in which a 
        monotonic relation is observed between standard deviation 
        and coverage of absolute value of error. Since the 
        statistics computed per bin are relatively noisy, the 
        application of a greedy criterion (e.g. guarantee a
        monotonically increasing relationship) does not yield 
        good results. Therefore, a softer version is constructed
        based on the satisfaction of certain criteria depending
        on: the values of the error coverage computed per bin,
        a smoothed version of them and the assocatiate error
        estimated (based on one standard deviation for a binomial
        distribution estimated by bin vs. the other bins).
        A minimal validation requiring the end idex to be
        largest than the starting index is performed before
        the function return.
        
        Current criteria:
        - the smoothed errors are inside the error bars AND
          they are almost increasing (a small tolerance is
          allowed, so a small wobbliness in the smoother
          values is permitted).
        OR
        - both the raw values for the bins (with a small tolerance)
          are increasing, AND the smoothed value is greater than the
          raw value.
        OR
        - the current smoothed value is greater than the previous AND 
          the smoothed values for the next been are inside the error
          bars.
        
        Parameters
        ----------
        error_thresholds : numpy array
            Thresholds of the errors computed to attain a certain
            error coverage per bin.
        error_thresholds_smooth : numpy array
            Thresholds of the errors computed to attain a certain
            error coverage per bin after a smoothed operation is applied
            to the frequently noisy bin-based estimations.
        err_err : numpy array
            Error bars in errors (one standard deviation for a binomial
            distribution estimated by bin vs. the other bins) for the 
            calibration error.
            
        Return
        ----------
        sigma_start_index : non-negative integer
            Index estimated in the mean_sigma array corresponing to
            the value that defines the start of the valid empirical 
            calibration interval (i.e. index to the smallest std for 
            which a meaningful error mapping is obtained, according
            to the criteria explained before).
        sigma_end_index : non-negative integer
            Index estimated in the mean_sigma array corresponing to
            the value that defines the end of the valid empirical
            calibration interval (i.e. index to the largest std for
            which a meaningful error mapping is obtained, according
            to the criteria explained before).
    """

    # Computation of the calibration interval
    limitH = error_thresholds + err_err
    limitL = error_thresholds - err_err

    # search for starting point
    for i in range(err_err.shape[0]):
        if ((error_thresholds_smooth[i] >= limitL[i]) and
         (error_thresholds_smooth[i] <= limitH[i])): # Ask if the current is in the interval
            sigma_start_index = i
            break
    sigma_end_index = sigma_start_index - 1

    restart = max(1, sigma_start_index)
    for i in range(restart, err_err.shape[0]-1):
        if (((error_thresholds_smooth[i] >= limitL[i]) and
            (error_thresholds_smooth[i] <= limitH[i]) and
            ((error_thresholds_smooth[i] * 1.005 > error_thresholds_smooth[i-1]) or
            ((error_thresholds[i] * 1.01 > error_thresholds[i-1]) and
            (error_thresholds_smooth[i] > error_thresholds[i])))) # Ask if the current is in the interval with slightly increasing trend
            or # Ask if the current is greater than the previous and the next is in the interval
            ((error_thresholds_smooth[i] > error_thresholds_smooth[i-1]) and
            ((error_thresholds_smooth[i+1] >= limitL[i+1]) and
            (error_thresholds_smooth[i+1] <= limitH[i+1])))):

            sigma_end_index = i
        else: # Finalize search for monotonic range
            if (sigma_end_index - sigma_start_index) > 4:
                break
            else: # Reset indices
                sigma_start_index = i + 1
                sigma_end_index = i

    print('Range of valid sigma indices (inclusive): %d --> %d' % (sigma_start_index, sigma_end_index))

    assert (sigma_end_index > sigma_start_index)

    return sigma_start_index, sigma_end_index


def applying_calibration(pSigma_test, pPred_test, true_test, s_interpolate, minL_sigma_auto, maxL_sigma_auto):
    """ Use the empirical mapping between standard deviation and
        absolute value of error estimated during calibration (i.e.
        apply the univariate spline computed) to estimate the error
        for the part of the standard deviation array that was reserved
        for testing the empirical calibration. The resulting error array
        (yp_test) should overestimate the true observed error (eabs_red).
        All the computations are restricted to the valid calibration
        interval: [minL_sigma_auto, maxL_sigma_auto].
        
        Parameters
        ----------
        pSigma_test : numpy array
            Part of the standard deviations array to use for calibration testing.
        pPred_test : numpy array
            Part of the predictions array to use for calibration testing.
        true_test : numpy array
            Part of the true (observed) values array to use for calibration testing.
        s_interpolate : scipy.interpolate python object
            A python object from scipy.interpolate that computes a
            univariate spline (InterpolatedUnivariateSpline) expressing
            the mapping from standard deviation to error. This
            spline is generated during the computational empirical
            calibration procedure.
        minL_sigma_auto : float
            Starting value of the valid empirical calibration interval
            (i.e. smallest std for which a meaningful error mapping
            is obtained).
        maxL_sigma_auto : float
            Ending value of the valid empirical calibration interval
            (i.e. largest std for which a meaningful error mappping
            is obtained).

        Return
        ----------
        index_sigma_range_test : numpy array
            Indices of the pSigma_test array that are included in the
            valid calibration interval, given by:
            [minL_sigma_auto, maxL_sigma_auto].
        xp_test : numpy array
            Array with the mean standard deviations in the calibration
            testing array.
        yp_test : numpy array
            Mapping of the given standard deviation to error computed
            from the interpolation spline constructed by empirical
            calibration.
        eabs_red : numpy array
            Array with the observed abolute errors in the part of the testing
            array for which the observed standard deviations are in the
            valid interval of calibration.
    """

    # Filter to appropriate range
    index_sigma_range_test = (pSigma_test >= minL_sigma_auto) & (pSigma_test < maxL_sigma_auto)
    xp_test = pSigma_test[index_sigma_range_test]
    yp_test = s_interpolate(xp_test)
    Er_vect_ = true_test  - pPred_test
    eabs_ = np.abs(Er_vect_)
    eabs_red = eabs_[index_sigma_range_test]

    return index_sigma_range_test, xp_test, yp_test, eabs_red


def overprediction_check(yp_test, eabs_red):
    """ Compute the percentage of overestimated absoulte error
        predictions for the arrays reserved for calibration testing
        and whose corresponding standard deviations are included
        in the valid calibration interval.
        
        Parameters
        ----------
        yp_test : numpy array
            Mapping of the standard deviation to error computed
            from the interpolation spline constructed by empirical
            calibration.
        eabs_red : numpy array
            Array with the observed abolute errors in the part of the testing
            array for which the observed standard deviations are in the
            valid interval of calibration.
    """

    over_pred_error_index =  (yp_test >= eabs_red)
    percentage_over_predicted = (over_pred_error_index.sum() / yp_test.shape[0])
    print("percentage over predicted: ", percentage_over_predicted)







