from __future__ import absolute_import

import numpy as np
from scipy.stats import pearsonr
from scipy import signal
from scipy import interpolate
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings


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

def compute_statistics_homoscedastic_summary(df_data,
                                            col_true=0,
                                            col_pred=6,
                                            col_std_pred=7,
                                            ):
    """ Extracts ground truth, mean prediction, error and
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
    pred_name = df_data.columns[col_true]
    Ypred = df_data.iloc[:,col_pred].values
    Ypred_std = df_data.iloc[:,col_std_pred].values
    yerror = Ytrue - Ypred
    sigma = Ypred_std # std
    MSE = mean_squared_error(Ytrue, Ypred_mean)
    print('MSE: ', MSE)
    MSE_STD = np.std((Ytrue - Ypred_mean)**2)
    print('MSE_STD: ', MSE_STD)
    MAE = mean_absolute_error(Ytrue, Ypred_mean)
    print('MAE: ', MAE)
    r2 = r2_score(Ytrue, Ypred_mean)
    print('R2: ', r2)
    # p-value 'not entirely reliable, reasonable for datasets > 500'
    pearson_cc, pval = pearsonr(Ytrue, Ypred_mean)
    print('Pearson CC: %f, p-value: %e' % (pearson_cc, pval))
    
    return Ytrue, Ypred, yerror, sigma, Ypred_std, pred_name


def compute_statistics_homoscedastic(df_data,
                                     col_true=4,
                                     col_pred_start=6
                                     ):
    """ Extracts ground truth, mean prediction, error and
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
    pred_name = df_data.columns[col_true]
    Ypred_mean_ = np.mean(df_data.iloc[:,col_pred_start:], axis=1)
    Ypred_mean = Ypred_mean_.values
    Ypred_std_ = np.std(df_data.iloc[:,col_pred_start:], axis=1)
    Ypred_std = Ypred_std_.values
    yerror = Ytrue - Ypred_mean
    sigma = Ypred_std # std
    MSE = mean_squared_error(Ytrue, Ypred_mean)
    print('MSE: ', MSE)
    MSE_STD = np.std((Ytrue - Ypred_mean)**2)
    print('MSE_STD: ', MSE_STD)
    MAE = mean_absolute_error(Ytrue, Ypred_mean)
    print('MAE: ', MAE)
    r2 = r2_score(Ytrue, Ypred_mean)
    print('R2: ', r2)
    # p-value 'not entirely reliable, reasonable for datasets > 500'
    pearson_cc, pval = pearsonr(Ytrue, Ypred_mean)
    print('Pearson CC: %f, p-value: %e' % (pearson_cc, pval))
        
    return Ytrue, Ypred_mean, yerror, sigma, Ypred_std, pred_name


def compute_statistics_heteroscedastic(df_data,
                                     col_true=4,
                                     col_pred_start=6,
                                     col_std_pred_start=7,
                                     ):
    """ Extracts ground truth, mean prediction, error, standard
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
            model. For heteroscedastic inference this corresponds to the
            sqrt(exp(s^2)) with s^2 predicted value.
        Ypred_std : numpy array
            Array with standard deviations computed from regular
            (homoscedastic) inference.
        pred_name : string
            Name of data colum or quantity predicted (as extracted
            from the data frame using the col_true index).
    """

    Ytrue = df_data.iloc[:,col_true].values
    pred_name = df_data.columns[col_true]
    Ypred_mean_ = np.mean(df_data.iloc[:,col_pred_start::2], axis=1)
    Ypred_mean = Ypred_mean_.values
    Ypred_std_ = np.std(df_data.iloc[:,col_pred_start::2], axis=1)
    Ypred_std = Ypred_std_.values
    yerror = Ytrue - Ypred_mean
    s_ = df_data.iloc[:,col_std_pred_start::2]
    s_mean = np.mean(s_, axis=1)
    var = np.exp(s_mean.values) # variance
    sigma = np.sqrt(var) # std
    MSE = mean_squared_error(Ytrue, Ypred_mean)
    print('MSE: ', MSE)
    MSE_STD = np.std((Ytrue - Ypred_mean)**2)
    print('MSE_STD: ', MSE_STD)
    MAE = mean_absolute_error(Ytrue, Ypred_mean)
    print('MAE: ', MAE)
    r2 = r2_score(Ytrue, Ypred_mean)
    print('R2: ', r2)
    # p-value 'not entirely reliable, reasonable for datasets > 500'
    pearson_cc, pval = pearsonr(Ytrue, Ypred_mean)
    print('Pearson CC: %f, p-value: %e' % (pearson_cc, pval))

    return Ytrue, Ypred_mean, yerror, sigma, Ypred_std, pred_name


def compute_statistics_quantile(df_data,
                                     sigma_divisor=2.56,
                                     col_true=4,
                                     col_pred_start=6
                                     ):
    """ Extracts ground truth, 50th percentile mean prediction,
        low percentile and high percentile mean prediction
        (usually 1st decile and 9th decile respectively),
        error (using 5th decile), standard deviation of
        prediction (using 5th decile) and predicted (learned)
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
            the difference between the 9th and 1st deciles).
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
            (usually the 1st decile).
        Ypred_Hp_mean : numpy array
            Array with predicted values of the higher percentile
            (usually the 9th decile).
    """

    Ytrue = df_data.iloc[:,col_true].values
    pred_name = df_data.columns[col_true]
    Ypred_5d_mean = np.mean(df_data.iloc[:,col_pred_start::3], axis=1)
    Ypred_mean = Ypred_5d_mean.values
    Ypred_Lp_mean_ = np.mean(df_data.iloc[:,col_pred_start+1::3], axis=1)
    Ypred_Hp_mean_ = np.mean(df_data.iloc[:,col_pred_start+2::3], axis=1)
    Ypred_Lp_mean = Ypred_Lp_mean_.values
    Ypred_Hp_mean = Ypred_Hp_mean_.values
    interdecile_range = Ypred_Hp_mean - Ypred_Lp_mean
    sigma = interdecile_range / sigma_divisor
    yerror = Ytrue - Ypred_mean
    Ypred_std_ = np.std(df_data.iloc[:,col_pred_start::3], axis=1)
    Ypred_std = Ypred_std_.values
    MSE = mean_squared_error(Ytrue, Ypred_mean)
    print('MSE: ', MSE)
    MSE_STD = np.std((Ytrue - Ypred_mean)**2)
    print('MSE_STD: ', MSE_STD)
    MAE = mean_absolute_error(Ytrue, Ypred_mean)
    print('MAE: ', MAE)
    r2 = r2_score(Ytrue, Ypred_mean)
    print('R2: ', r2)
    # p-value 'not entirely reliable, reasonable for datasets > 500'
    pearson_cc, pval = pearsonr(Ytrue, Ypred_mean)
    print('Pearson CC: %f, p-value: %e' % (pearson_cc, pval))

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



def compute_empirical_calibration_interpolation(pSigma_cal, pPred_cal, true_cal, cv=10):
    """ Use the arrays provided to estimate an empirical mapping
        between standard deviation and absolute value of error,
        both of which have been observed during inference. Since
        most of the times the prediction statistics are very noisy,
        two smoothing steps (based on scipy's savgol filter) are performed.
        Cubic Hermite splines (PchipInterpolator) are constructed for
        interpolation. This type of splines preserves the monotonicity
        in the interpolation data and does not overshoot if the data is
        not smooth. The overall process of constructing a spline
        to express the mapping from standard deviation to error is
        composed of smoothing-interpolation-smoothing-interpolation.
        
        Parameters
        ----------
        pSigma_cal : numpy array
            Part of the standard deviations array to use for calibration.
        pPred_cal : numpy array
            Part of the predictions array to use for calibration.
        true_cal : numpy array
            Part of the true (observed) values array to use for calibration.
        cv : int
            Number of cross validations folds to run to determine a 'good'
            fit.
            
        Return
        ----------
        splineobj_best : scipy.interpolate python object
            A python object from scipy.interpolate that computes a
            cubic Hermite splines (PchipInterpolator) constructed
            to express the mapping from standard deviation to error after a
            'drastic' smoothing of the predictions. A 'good' fit is
            determined by taking the spline for the fold that produces
            the smaller mean absolute error in testing data (not used
            for the smoothing / interpolation).
        splineobj2 : scipy.interpolate python object
            A python object from scipy.interpolate that computes a
            cubic Hermite splines (PchipInterpolator) constructed
            to express the mapping from standard deviation to error. This
            spline is generated for interpolating the samples generated
            after the smoothing of the first interpolation spline (i.e.
            splineobj_best).
    """

    xs3 = pSigma_cal # std
    z3 = np.abs(true_cal - pPred_cal) # abs error

    test_split = 1.0 / cv
    xmin = np.min(pSigma_cal)
    xmax = np.max(pSigma_cal)
    
    warnings.filterwarnings("ignore")
    
    print('--------------------------------------------')
    print('Using CV for selecting calibration smoothing')
    print('--------------------------------------------')

    min_error = np.inf
    for cv_ in range(cv):
        # Split data for the different folds
        X_train, X_test, y_train, y_test = train_test_split(xs3, z3, test_size=test_split, shuffle=True)
            
        # Order x to apply smoothing and interpolation
        xindsort = np.argsort(X_train)
        # Smooth abs error
        #z3smooth = signal.savgol_filter(y_train[xindsort], 31, 1, mode='nearest')
        z3smooth = signal.savgol_filter(y_train[xindsort], 21, 1, mode='nearest')
        # Compute Piecewise Cubic Hermite Interpolating Polynomial
        splineobj = interpolate.PchipInterpolator(X_train[xindsort], z3smooth, extrapolate=True)
        # Compute prediction from interpolator
        ytest = splineobj(X_test)
        # compute MAE of true ABS error vs predicted ABS error
        mae = mean_absolute_error(y_test, ytest)
        print('MAE: ', mae)

        if mae < min_error: # store spline interpolator for best fold
            min_error = mae
            splineobj_best = splineobj
            
    # Smooth again
    xp23 = np.linspace(xmin, xmax, 200)
    # Predict using best interpolator from folds
    yp23 = splineobj_best(xp23)
    # Smooth the ABS error predicted
    yp23smooth = signal.savgol_filter(yp23, 15, 1, mode='nearest')
    # Compute spline over second level of smoothing
    splineobj2 = interpolate.PchipInterpolator(xp23, yp23smooth, extrapolate=True)

    return splineobj_best, splineobj2
