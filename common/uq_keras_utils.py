from __future__ import absolute_import

from keras import backend as K

from keras.callbacks import Callback

from keras.models import Model
from keras.layers import Dense
from keras import layers

from keras.utils import np_utils
from keras.metrics import mean_squared_error, mean_absolute_error

import numpy as np

from scipy.stats import norm, cauchy

piSQ = np.pi**2

###################################################################

# For Abstention Model


def abstention_loss(alpha, mask):
    """ Function to compute abstention loss.
        It is composed by two terms:
        (i) original loss of the multiclass classification problem,
        (ii) cost associated to the abstaining samples.

    Parameters
    ----------
    alpha : Keras variable
        Weight of abstention term in cost function
    mask : ndarray
        Numpy array to use as mask for abstention:
        it is 1 on the output associated to the abstention class and 0 otherwise
    """

    def loss(y_true, y_pred):
        """
        Parameters
        ----------
        y_true : keras tensor
            True values to predict
        y_pred : keras tensor
            Prediction made by the model.
            It is assumed that this keras tensor includes extra columns to store the abstaining classes.
        """
        base_pred = (1 - mask) * y_pred + K.epsilon()
        base_true = y_true
        base_cost = K.categorical_crossentropy(base_true, base_pred)
        #abs_pred = K.mean(mask * y_pred, axis=-1)
        abs_pred = K.sum(mask * y_pred, axis=-1)
        # add some small value to prevent NaN when prediction is abstained
        abs_pred = K.clip(abs_pred, K.epsilon(), 1.-K.epsilon())

        #return ((1. - abs_pred) * base_cost - alpha * K.log(1. - abs_pred))
        return K.mean((1. - abs_pred) * base_cost - alpha * K.log(1. - abs_pred))

    loss.__name__ = 'abs_crossentropy'
    return loss


def sparse_abstention_loss(alpha, mask):
    """ Function to compute abstention loss.
        It is composed by two terms:
        (i) original loss of the multiclass classification problem,
        (ii) cost associated to the abstaining samples.
        Assumes y_true is not one-hot encoded.

    Parameters
    ----------
    alpha : Keras variable
        Weight of abstention term in cost function
    mask : ndarray
        Numpy array to use as mask for abstention: it is 1 on the output associated to the abstention class and 0 otherwise
    """
    def loss(y_true, y_pred):
        """
        Parameters
        ----------
        y_true : keras tensor
            True values to predict
        y_pred : keras tensor
            Prediction made by the model. It is assumed that this keras tensor includes extra columns to store the abstaining classes.
        """
        base_pred = (1 - mask) * y_pred + K.epsilon()
        base_true = y_true
        base_cost = K.sparse_categorical_crossentropy(base_true, base_pred)
        #abs_pred = K.mean(mask * y_pred, axis=-1)
        abs_pred = K.sum(mask * y_pred, axis=-1)
        # add some small value to prevent NaN when prediction is abstained
        abs_pred = K.clip(abs_pred, K.epsilon(), 1.-K.epsilon())

        #return ((1. - abs_pred) * base_cost - alpha * K.log(1. - abs_pred))
        return K.mean((1. - abs_pred) * base_cost - alpha * K.log(1. - abs_pred))

    loss.__name__ = 'sparse_abs_crossentropy'
    return loss


def abstention_acc_metric(nb_classes):
    """ Abstained accuracy:
        Function to estimate accuracy over the predicted samples
        after removing the samples where the model is abstaining.

    Parameters
    ----------
    nb_classes : int or ndarray
        Integer or numpy array defining indices of the abstention class
    """
    def metric(y_true, y_pred):
        """
        Parameters
        ----------
        y_true : keras tensor
            True values to predict
        y_pred : keras tensor
            Prediction made by the model.
            It is assumed that this keras tensor includes extra columns to store the abstaining classes.
        """
        # matching in original classes
        true_pred = K.sum(K.cast(K.equal(K.argmax(y_true, axis=-1), K.argmax(y_pred, axis=-1)), 'int64'))

        # total abstention
        total_abs = K.sum(K.cast(K.equal(K.argmax(y_pred, axis=-1), nb_classes), 'int64'))

        # total predicted in original classes
        total_pred = K.sum(K.cast(K.equal(K.argmax(y_pred, axis=-1), K.argmax(y_pred, axis=-1)), 'int64'))

        # guard against divide by zero
        condition = K.greater(total_pred, total_abs)
        abs_acc = K.switch(condition, true_pred/(total_pred - total_abs), total_pred/total_pred)
        return abs_acc

    metric.__name__ = 'abstention_acc'
    return metric


def sparse_abstention_acc_metric(nb_classes):
    """ Abstained accuracy:
        Function to estimate accuracy over the predicted samples
        after removing the samples where the model is abstaining.
        Assumes y_true is not one-hot encoded.

    Parameters
    ----------
    nb_classes : int or ndarray
        Integer or numpy array defining indices of the abstention class
    """
    def metric(y_true, y_pred):
        """
        Parameters
        ----------
        y_true : keras tensor
            True values to predict
        y_pred : keras tensor
            Prediction made by the model.
            It is assumed that this keras tensor includes extra columns to store the abstaining classes.
        """
        # matching in original classes
        y_pred_index = K.argmax(y_pred, axis=-1)
        y_true_index = K.cast(K.max(y_true, axis=-1), 'int64')
        true_pred = K.sum(K.cast(K.equal(y_true_index, y_pred_index), 'int64'))

        # total abstention
        total_abs = K.sum(K.cast(K.equal(K.argmax(y_pred, axis=-1), nb_classes), 'int64'))

        # total predicted in original classes
        total_pred = K.sum(K.cast(K.equal(K.argmax(y_pred, axis=-1), K.argmax(y_pred, axis=-1)), 'int64'))

        # guard against divide by zero
        condition = K.greater(total_pred, total_abs)
        abs_acc = K.switch(condition, true_pred/(total_pred - total_abs), total_pred/total_pred)
        return abs_acc

    metric.__name__ = 'sparse_abstention_acc'
    return metric


def abstention_metric(nb_classes):
    """ Function to estimate fraction of the samples where the model is abstaining.

    Parameters
    ----------
    nb_classes : int or ndarray
        Integer or numpy array defining indices of the abstention class
    """
    def metric(y_true, y_pred):
        """
        Parameters
        ----------
        y_true : keras tensor
            True values to predict
        y_pred : keras tensor
            Prediction made by the model.
            It is assumed that this keras tensor includes extra columns to store the abstaining classes.
        """
        # total abstention
        total_abs = K.sum(K.cast(K.equal(K.argmax(y_pred, axis=-1), nb_classes), 'int64'))

        # total predicted in original classes
        total_pred = K.sum(K.cast(K.equal(K.argmax(y_pred, axis=-1), K.argmax(y_pred, axis=-1)), 'int64'))

        return total_abs / total_pred

    metric.__name__ = 'abstention'
    return metric


def acc_class_i_metric(class_i):
    """ Function to estimate accuracy over the ith class prediction.
        This estimation is global (i.e. abstaining samples are not removed)

    Parameters
    ----------
    class_i : int
        Index of the class to estimate accuracy
    """
    def metric(y_true, y_pred):
        """
        Parameters
        ----------
        y_true : keras tensor
            True values to predict
        y_pred : keras tensor
            Prediction made by the model.
            It is assumed that this keras tensor includes extra columns to store the abstaining classes.
        """
        # Find locations in ground truth belonging to class i
        ytrueint = K.cast(K.equal(K.argmax(y_true, axis=-1), class_i), 'int64')

        # Compute total number of ground truth samples in class i
        total_true_i = K.sum(ytrueint)

        # Find samples in prediction belonging to class i (not accounting for abstention)
        ypredint = K.cast(K.equal(K.argmax(y_pred[:, :-1], axis=-1), class_i), 'int64')

        # Find correctly predicted class i samples
        true_i_pred = K.sum(ytrueint * ypredint)

        # Compute accuracy in class i
        acc = true_i_pred / total_true_i

        # Since there could be few samples in class i
        # it is possible that ground truth does not
        # have any sample in class i, leading to a divide
        # by zero and not valid accuracy
        # Therefore, for the accuracy to be valid
        # total_true_i should be greater than zero
        # otherwise, return 0.

        condition = K.greater(total_true_i, 0)

        return K.switch(condition, acc, K.zeros_like(acc, dtype=acc.dtype))

    metric.__name__ = 'acc_class_{}'.format(class_i)
    return metric


def abstention_acc_class_i_metric(nb_classes, class_i):
    """ Function to estimate accuracy over the class i prediction after removing the samples where the model is abstaining.

    Parameters
    ----------
    nb_classes : int or ndarray
        Integer or numpy array defining indices of the abstention class
    class_i : int
        Index of the class to estimate accuracy after removing abstention samples
    """
    def metric(y_true, y_pred):
        """
        Parameters
        ----------
        y_true : keras tensor
            True values to predict
        y_pred : keras tensor
            Prediction made by the model.
            It is assumed that this keras tensor includes extra columns to store the abstaining classes.
        """
        # Find locations in ground truth belonging to class i
        ytrueint = K.cast(K.equal(K.argmax(y_true, axis=-1), class_i), 'int64')

        # Find locations that are predicted (not abstained)
        mask_pred = K.cast(K.not_equal(K.argmax(y_pred, axis=-1), nb_classes), 'int64')

        # Compute total number of ground truth samples in class i filtering abstaining predictions
        total_true_i = K.sum(ytrueint * mask_pred)

        # matching in original class i after removing abstention
        true_i_pred = K.sum(mask_pred * ytrueint * K.cast(K.equal(K.argmax(y_true, axis=-1), K.argmax(y_pred, axis=-1)), 'int64'))

        # Compute accuracy in class i
        acc = true_i_pred / total_true_i

        # Since there could few samples in class i
        # it is possible that ground truth does not
        # have any sample in class i, leading to a divide
        # by zero and not valid accuracy
        # Therefore, for the accuracy to be valid
        # total_true_i should be greater than zero
        # otherwise, return 0.

        condition = K.greater(total_true_i, 0)

        return K.switch(condition, acc, K.zeros_like(acc, dtype=acc.dtype))

    metric.__name__ = 'abstention_acc_class_{}'.format(class_i)
    return metric


def abstention_class_i_metric(nb_classes, class_i):
    """ Function to estimate fraction of the samples where the model is abstaining in class i.

    Parameters
    ----------
    nb_classes : int or ndarray
        Integer or numpy array defining indices of the abstention class
    class_i : int
        Index of the class to estimate accuracy
    """
    def metric(y_true, y_pred):
        """
        Parameters
        ----------
        y_true : keras tensor
            True values to predict
        y_pred : keras tensor
            Prediction made by the model. It is assumed that this keras tensor includes extra columns to store the abstaining classes.
        """
        # Find locations in ground truth belonging to class i
        ytrue_i_int = K.cast(K.equal(K.argmax(y_true, axis=-1), class_i), 'int64')
        # total in class i
        total_class_i = K.sum(ytrue_i_int)

        # Abstention samples
        y_abs = K.cast(K.equal(K.argmax(y_pred, axis=-1), nb_classes), 'int64')
        # Abstention in class_i
        total_abs_i = K.sum(ytrue_i_int * y_abs)

        abs_i = total_abs_i / total_class_i

        condition = K.greater(total_class_i, 0)

        return K.switch(condition, abs_i, K.zeros_like(abs_i, dtype=abs_i.dtype))

    metric.__name__ = 'abstention_class_{}'.format(class_i)
    return metric


class AbstentionAdapt_Callback(Callback):
    """ This callback is used to adapt the parameter alpha in the abstention loss.
        The parameter alpha (weight of the abstention term in the abstention loss) is increased or decreased adaptively during the training run. It is decreased if the current abstention accuracy is less than the minimum accuracy set or increased if the current abstention fraction is greater than the maximum fraction set.
        The abstention accuracy metric to use must be specified as the 'acc_monitor' argument in the initialization of the callback. It could be: the global abstention accuracy (abstention_acc), the abstention accuracy over the ith class (acc_class_i), etc.
        The abstention metric to use must be specified as the 'abs_monitor' argument in the initialization of the callback. It should be the metric that computes the fraction of samples for which the model is abstaining (abstention).
        The factor alpha is modified if the current abstention accuracy is less than the minimum accuracy set or if the current abstention fraction is greater than the maximum fraction set. Thresholds for minimum and maximum correction factors are computed and the correction over alpha is not allowed to be less or greater than them, respectively, to avoid huge swings in the abstention loss evolution.
    """

    def __init__(self, acc_monitor, abs_monitor, alpha0, init_abs_epoch=4, alpha_scale_factor=0.8, min_abs_acc=0.9, max_abs_frac=0.4, acc_gain=5.0, abs_gain=1.0):
        """ Initializer of the AbstentionAdapt_Callback.
        Parameters
        ----------
        acc_monitor : keras metric
            Accuracy metric to monitor during the run and use as base to adapt the weight of the abstention term (i.e. alpha) in the abstention cost function. (Must be an accuracy metric that takes abstention into account).
        abs_monitor : keras metric
            Abstention metric monitored during the run and used as the other factor to adapt the weight of the abstention term (i.e. alpha) in the asbstention loss function
        alpha0 : float
            Initial weight of abstention term in cost function
        init_abs_epoch : integer
            Value of the epochs to start adjusting the weight of the abstention term (i.e. alpha). Default: 4.
        alpha_scale_factor: float
            Factor to scale (increase by dividing or decrease by multiplying) the weight of the abstention term (i.e. alpha). Default: 0.8.
        min_abs_acc: float
            Minimum accuracy to target in the current training. Default: 0.9.
        max_abs_frac: float
            Maximum abstention fraction to tolerate in the current training. Default: 0.4.
        acc_gain: float
            Factor to adjust alpha scale. Default: 5.0.
        abs_gain: float
            Factor to adjust alpha scale. Default: 1.0.
        """
        super(AbstentionAdapt_Callback, self).__init__()

        self.acc_monitor = acc_monitor  # Keras metric to monitor (must be an accuracy with abstention)
        self.abs_monitor = abs_monitor  # Keras metric momitoring abstention fraction
        self.alpha = K.variable(value=alpha0)     # Weight of abstention term
        self.init_abs_epoch = init_abs_epoch  # epoch to init abstention
        self.alpha_scale_factor = alpha_scale_factor  # factor to scale alpha (weight for abstention term in cost function)
        self.min_abs_acc = min_abs_acc  # minimum target accuracy (value specified as parameter of the run)
        self.max_abs_frac = max_abs_frac  # maximum abstention fraction (value specified as parameter of the run)
        self.acc_gain = acc_gain  # factor for adjusting alpha scale
        self.abs_gain = abs_gain  # factor for adjusting alpha scale
        self.alphavalues = []  # array to store alpha evolution

    def on_epoch_end(self, epoch, logs=None):
        """ Updates the weight of abstention term on epoch end.
        Parameters
        ----------
        epoch : integer
            Current epoch in training.
        logs : keras logs
            Metrics stored during current keras training.
        """

        new_alpha_val = K.get_value(self.alpha)
        if epoch > self.init_abs_epoch:
            if self.acc_monitor is None or self.abs_monitor is None:
                raise Exception('ERROR! Abstention Adapt conditioned on metrics ' + str(self.acc_monitor) + ' and ' + str(self.abs_monitor) + ' which are not available. Available metrics are: ' + ','.join(list(logs.keys())) + '... Exiting')
            else:
                # Current accuracy (with abstention)
                abs_acc = logs.get(self.acc_monitor)
                # Current abstention fraction
                abs_frac = logs.get(self.abs_monitor)
                if abs_acc is None or abs_frac is None:
                    raise Exception('ERROR! Abstention Adapt conditioned on metrics ' + str(self.acc_monitor) + ' and ' + str(self.abs_monitor) + ' which are not available. Available metrics are: ' + ','.join(list(logs.keys())) + '... Exiting')

                # modify alpha as needed
                acc_error = abs_acc - self.min_abs_acc
                acc_error = min(acc_error, 0.0)
                abs_error = abs_frac - self.max_abs_frac
                abs_error = max(abs_error, 0.0)
                new_scale = 1.0 + self.acc_gain * acc_error + self.abs_gain * abs_error
                # threshold to avoid huge swings
                min_scale = self.alpha_scale_factor
                max_scale = 1. / self.alpha_scale_factor
                new_scale = min(new_scale, max_scale)
                new_scale = max(new_scale, min_scale)

                #print('Scaling factor: ', new_scale)
                new_alpha_val *= new_scale
                K.set_value(self.alpha, new_alpha_val)
                print('Scaling factor: ', new_scale, ' new alpha, ', new_alpha_val)

        self.alphavalues.append(new_alpha_val)


def modify_labels(numclasses_out, ytrain, ytest, yval=None):
    """ This function generates a categorical representation with a class added for indicating abstention.

    Parameters
    ----------
    numclasses_out : integer
        Original number of classes + 1 abstention class
    ytrain : ndarray
        Numpy array of the classes (labels) in the training set
    ytest : ndarray
        Numpy array of the classes (labels) in the testing set
    yval : ndarray
        Numpy array of the classes (labels) in the validation set
    """

    classestrain = np.max(ytrain) + 1
    classestest = np.max(ytest) + 1
    if yval is not None:
        classesval = np.max(yval) + 1

    assert(classestrain == classestest)
    if yval is not None:
        assert(classesval == classestest)
    assert((classestrain+1) == numclasses_out)  # In this case only one other slot for abstention is created

    labels_train = np_utils.to_categorical(ytrain, numclasses_out)
    labels_test = np_utils.to_categorical(ytest, numclasses_out)
    if yval is not None:
        labels_val = np_utils.to_categorical(yval, numclasses_out)

    # For sanity check
    mask_vec = np.zeros(labels_train.shape)
    mask_vec[:, -1] = 1
    i = np.random.choice(range(labels_train.shape[0]))
    sanity_check = mask_vec[i, :]*labels_train[i, :]
    print(sanity_check.shape)
    if ytrain.ndim > 1:
        ll = ytrain.shape[1]
    else:
        ll = 0

    for i in range(ll):
        for j in range(numclasses_out):
            if sanity_check[i, j] == 1:
                print('Problem at ', i, j)

    if yval is not None:
        return labels_train, labels_test, labels_val

    return labels_train, labels_test

###################################################################


def add_model_output(modelIn, mode=None, num_add=None, activation=None):
    """ This function modifies the last dense layer in the passed keras model. The modification includes adding units and optionally changing the activation function.

    Parameters
    ----------
    modelIn : keras model
        Keras model to be modified.
    mode : string
        Mode to modify the layer. It could be:
        'abstain' for adding an arbitrary number of units for the abstention optimization strategy.
        'qtl' for quantile regression which needs the outputs to be tripled.
        'het' for heteroscedastic regression which needs the outputs to be doubled.
    num_add : integer
        Number of units to add. This only applies to the 'abstain' mode.
    activation : string
        String with keras specification of activation function (e.g. 'relu', 'sigomid', 'softmax', etc.)

    Return
    ----------
    modelOut : keras model
        Keras model after last dense layer has been modified as specified. If there is no mode specified it returns the same model. If the mode is not one of 'abstain', 'qtl' or 'het' an exception is raised.
    """

    if mode is None:
        return modelIn

    numlayers = len(modelIn.layers)
    # Find last dense layer
    i = -1
    while 'dense' not in (modelIn.layers[i].name) and ((i+numlayers) > 0):
        i -= 1
    # Minimal verification about the validity of the layer found
    assert ((i + numlayers) >= 0)
    assert ('dense' in modelIn.layers[i].name)

    # Compute new output size
    if mode == 'abstain':
        assert num_add is not None
        new_output_size = modelIn.layers[i].output_shape[-1] + num_add
    elif mode == 'qtl':  # for quantile UQ
        new_output_size = 3 * modelIn.layers[i].output_shape[-1]
    elif mode == 'het':  # for heteroscedastic UQ
        new_output_size = 2 * modelIn.layers[i].output_shape[-1]
    else:
        raise Exception('ERROR ! Type of mode specified ' \
                        + 'for adding outputs to the model: ' \
                        + mode + ' not implemented... Exiting')

    # Recover current layer options
    config = modelIn.layers[i].get_config()
    # Update number of units
    config['units'] = new_output_size
    # Update activation function if requested
    if activation is not None:
        config['activation'] = activation
    # Bias initialization seems to help het and qtl
    if mode == 'het' or mode == 'qtl':
        config['bias_initializer'] = 'ones'
    # Create new Dense layer
    reconstructed_layer = Dense.from_config(config)
    # Connect new Dense last layer to previous one-before-last layer
    additional = reconstructed_layer(modelIn.layers[i-1].output)
    # If the layer to replace is not the last layer, add the remainder layers
    if i < -1:
        for j in range(i+1, 0):
            config_j = modelIn.layers[j].get_config()
            aux_j = layers.deserialize({'class_name': modelIn.layers[j].__class__.__name__,
                                        'config': config_j})
            reconstructed_layer = aux_j.from_config(config_j)
            additional = reconstructed_layer(additional)

    modelOut = Model(modelIn.input, additional)

    return modelOut

###################################################################

# UQ regression - utilities


def r2_heteroscedastic_metric(nout):
    """This function computes the r2 for the heteroscedastic model. The r2 is computed over the prediction of the mean and the standard deviation prediction is not taken into account.

    Parameters
    ----------
    nout : int
        Number of outputs without uq augmentation
    """
    def metric(y_true, y_pred):
        """
        Parameters
        ----------
        y_true : Keras tensor
            Keras tensor including the ground truth
        y_pred : Keras tensor
            Keras tensor including the predictions of a heteroscedastic model. The predictions follow the order: (mean_0, S_0, mean_1, S_1, ...) with S_i the log of the variance for the ith output.
        """

        if nout > 1:
            y_out = K.reshape(y_pred[:, 0::nout], K.shape(y_true))
        else:
            y_out = K.reshape(y_pred[:, 0], K.shape(y_true))

        SS_res = K.sum(K.square(y_true - y_out))
        SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
        return (1. - SS_res/(SS_tot + K.epsilon()))

    metric.__name__ = 'r2_heteroscedastic'
    return metric


def mae_heteroscedastic_metric(nout):
    """This function computes the mean absolute error (mae) for the heteroscedastic model. The mae is computed over the prediction of the mean and the standard deviation prediction is not taken into account.

    Parameters
    ----------
    nout : int
        Number of outputs without uq augmentation
    """
    def metric(y_true, y_pred):
        """
        Parameters
        ----------
        y_true : Keras tensor
            Keras tensor including the ground truth
        y_pred : Keras tensor
            Keras tensor including the predictions of a heteroscedastic model. The predictions follow the order: (mean_0, S_0, mean_1, S_1, ...) with S_i the log of the variance for the ith output.
        """
        if nout > 1:
            y_out = K.reshape(y_pred[:, 0::nout], K.shape(y_true))
        else:
            y_out = K.reshape(y_pred[:, 0], K.shape(y_true))

        return mean_absolute_error(y_true, y_out)

    metric.__name__ = 'mae_heteroscedastic'
    return metric


def mse_heteroscedastic_metric(nout):
    """This function computes the mean squared error (mse) for the heteroscedastic model. The mse is computed over the prediction of the mean and the standard deviation prediction is not taken into account.

    Parameters
    ----------
    nout : int
        Number of outputs without uq augmentation
    """
    def metric(y_true, y_pred):
        """
        Parameters
        ----------
        y_true : Keras tensor
            Keras tensor including the ground truth
        y_pred : Keras tensor
            Keras tensor including the predictions of a heteroscedastic model. The predictions follow the order: (mean_0, S_0, mean_1, S_1, ...) with S_i the log of the variance for the ith output.
        """
        if nout > 1:
            y_out = K.reshape(y_pred[:, 0::nout], K.shape(y_true))
        else:
            y_out = K.reshape(y_pred[:, 0], K.shape(y_true))

        return mean_squared_error(y_true, y_out)

    metric.__name__ = 'mse_heteroscedastic'
    return metric


def meanS_heteroscedastic_metric(nout):
    """This function computes the mean log of the variance (log S) for the heteroscedastic model. The mean log is computed over the standard deviation prediction and the mean prediction is not taken into account.

    Parameters
    ----------
    nout : int
        Number of outputs without uq augmentation
    """
    def metric(y_true, y_pred):
        """
        Parameters
        ----------
        y_true : Keras tensor
            Keras tensor including the ground truth
        y_pred : Keras tensor
            Keras tensor including the predictions of a heteroscedastic model. The predictions follow the order: (mean_0, S_0, mean_1, S_1, ...) with S_i the log of the variance for the ith output.
        """
        if nout > 1:
            log_sig2 = y_pred[:, 1::nout]
        else:
            log_sig2 = y_pred[:, 1]

        return K.mean(log_sig2)

    metric.__name__ = 'meanS_heteroscedastic'
    return metric


def heteroscedastic_loss(nout):
    """This function computes the heteroscedastic loss for the heteroscedastic model. Both mean and standard deviation predictions are taken into account.

    Parameters
    ----------
    nout : int
        Number of outputs without uq augmentation
    """
    def loss(y_true, y_pred):
        """This function computes the heteroscedastic loss.

        Parameters
        ----------
        y_true : Keras tensor
            Keras tensor including the ground truth
        y_pred : Keras tensor
            Keras tensor including the predictions of a heteroscedastic model. The predictions follow the order: (mean_0, S_0, mean_1, S_1, ...) with S_i the log of the variance for the ith output.
        """
        y_shape = K.shape(y_true)

        if nout > 1:
            y_out = K.reshape(y_pred[:, 0::nout], y_shape)
            log_sig2 = K.reshape(y_pred[:, 1::nout], y_shape)
        else:
            y_out = K.reshape(y_pred[:, 0], y_shape)
            log_sig2 = K.reshape(y_pred[:, 1], y_shape)

        diff_sq = K.square(y_out - y_true)

        return K.mean(K.exp(-log_sig2) * diff_sq + log_sig2)

    # Return a function
    return loss


def quantile_loss(quantile, y_true, y_pred):
    """This function computes the quantile loss for a given quantile fraction.

    Parameters
    ----------
    quantile : float in (0, 1)
        Quantile fraction to compute the loss.
    y_true : Keras tensor
        Keras tensor including the ground truth
    y_pred : Keras tensor
        Keras tensor including the predictions of a quantile model.
    """

    error = (y_true - y_pred)
    return K.mean(K.maximum(quantile*error, (quantile-1)*error))


def triple_quantile_loss(nout, lowquantile, highquantile):
    """This function computes the quantile loss for the median and low and high quantiles. The median is given twice the weight of the other components.

    Parameters
    ----------
    nout : int
        Number of outputs without uq augmentation
    lowquantile: float in (0, 1)
        Fraction corresponding to the low quantile
    highquantile: float in (0, 1)
        Fraction corresponding to the high quantile
    """
    def loss(y_true, y_pred):
        """This function computes the quantile loss, considering the median and low and high quantiles.

        Parameters
        ----------
        y_true : Keras tensor
            Keras tensor including the ground truth
        y_pred : Keras tensor
            Keras tensor including the predictions of a heteroscedastic model. The predictions follow the order: (q50_0, qlow_0, qhigh_0, q50_1, qlow_1, qhigh_1, ...) with q50_i the median of the ith output and qlow_i and qhigh_i the low and high specified quantiles of the ith output.
        """

        y_shape = K.shape(y_true)
        if nout > 1:
            y_out0 = K.reshape(y_pred[:, 0::nout], y_shape)
            y_out1 = K.reshape(y_pred[:, 1::nout], y_shape)
            y_out2 = K.reshape(y_pred[:, 2::nout], y_shape)
        else:
            y_out0 = K.reshape(y_pred[:, 0], y_shape)
            y_out1 = K.reshape(y_pred[:, 1], y_shape)
            y_out2 = K.reshape(y_pred[:, 2], y_shape)

        return quantile_loss(lowquantile, y_true, y_out1) + quantile_loss(highquantile, y_true, y_out2) + 2. * quantile_loss(0.5, y_true, y_out0)

    return loss


def quantile_metric(nout, index, quantile):
    """This function computes the quantile metric for a given quantile and corresponding output index. This is provided as a metric to track evolution while training.

    Parameters
    ----------
    nout : int
        Number of outputs without uq augmentation
    index : int
        Index of output corresponding to the given quantile.
    quantile: float in (0, 1)
        Fraction corresponding to the quantile
    """
    def metric(y_true, y_pred):
        """
        Parameters
        ----------
        y_true : Keras tensor
            Keras tensor including the ground truth
        y_pred : Keras tensor
            Keras tensor including the predictions of a quantile model. The predictions follow the order: (q50_0, qlow_0, qhigh_0, q50_1, qlow_1, qhigh_1, ...) with q50_i the median of the ith output and qlow_i and qhigh_i the low and high specified quantiles of the ith output.
        """
        y_shape = K.shape(y_true)
        if nout > 1:
            y_out = K.reshape(y_pred[:, index::nout], y_shape)
        else:
            y_out = K.reshape(y_pred[:, index], y_shape)
        return quantile_loss(quantile, y_true, y_out)

    metric.__name__ = 'quantile_{}'.format(quantile)
    return metric


###################################################################

# For the Contamination Model


def add_index_to_output(y_train):
    """ This function adds a column to the training output to store the indices of the corresponding samples in the training set.

    Parameters
    ----------
    y_train : ndarray
        Numpy array of the output in the training set
    """
    # Add indices to y
    y_train_index = range(y_train.shape[0])
    y_train_augmented = np.vstack([y_train, y_train_index]).T

    return y_train_augmented


def contamination_loss(nout, T_k, a, sigmaSQ, gammaSQ):
    """ Function to compute contamination loss. It is composed by two terms: (i) the loss with respect to the normal distribution that models the distribution of the training data samples, (ii) the loss with respect to the Cauchy distribution that models the distribution of the outlier samples. Note that the evaluation of this contamination loss function does not make sense for any data different to the training set. This is because latent variables are only defined for samples in the training set.

    Parameters
    ----------
    nout : int
        Number of outputs without uq augmentation (in the contamination model the augmentation corresponds to the data index in training).
    T_k : Keras tensor
        Tensor containing latent variables (probability of membership to normal and Cauchy distributions) for each of the samples in the training set. (Validation data is usually augmented too to be able to run training with validation set, however loss in validation should not be used as a criterion for early stopping training since the latent variables are defined for the training only, and thus, are not valid when used in combination with data different from training).
    a : Keras variable
        Probability of belonging to the normal distribution
    sigmaSQ : Keras variable
        Variance estimated for the normal distribution
    gammaSQ : Keras variable
        Scale estimated for the Cauchy distribution
    """
    def loss(y_true, y_pred):
        """
        Parameters
        ----------
        y_true : keras tensor
            True values to predict. It is assumed that this keras tensor includes extra columns to store the index of the data sample in the training set.
        y_pred : keras tensor
            Prediction made by the model.
        """
        y_shape = K.shape(y_pred)
        y_true_ = K.reshape(y_true[:, :-1], y_shape)
        if nout > 1:
            diff_sq = K.sum(K.square(y_true_ - y_pred), axis=-1)
        else:
            diff_sq = K.square(y_true_ - y_pred)

        term_normal = diff_sq / (2. * sigmaSQ) + 0.5 * K.log(sigmaSQ) + 0.5 * K.log(2. * np.pi) - K.log(a)
        term_cauchy = K.log(1. + diff_sq / gammaSQ) + 0.5 * K.log(piSQ * gammaSQ) - K.log(1. - a)

        batch_index = K.cast(y_true[:, -1], 'int64')

        T_0_red = K.gather(T_k[:, 0], batch_index)
        T_1_red = K.gather(T_k[:, 1], batch_index)

        return K.sum(T_0_red * term_normal + T_1_red * term_cauchy)

    return loss


class Contamination_Callback(Callback):
    """ This callback is used to update the parameters of the contamination model. This functionality follows the EM algorithm: in the E-step latent variables are updated and in the M-step global variables are updated. The global variables correspond to 'a' (probability of membership to normal class), 'sigmaSQ' (variance of normal class) and 'gammaSQ' (scale of Cauchy class, modeling outliers). The latent variables correspond to 'T_k' (the first column corresponds to the probability of membership to the normal distribution, while the second column corresponds to the probability of membership to the Cauchy distribution i.e. outlier).
    """
    def __init__(self, x, y, a_max=0.99):
        """ Initializer of the Contamination_Callback.
        Parameters
        ----------
        x : ndarray
            Array of samples (= input features) in training set.
        y : ndarray
            Array of sample outputs in training set.
        a_max : float
            Maximum value of a variable to allow
        """
        super(Contamination_Callback, self).__init__()

        self.x = x                              # Features of training set
        self.y = y                              # Output of training set
        self.a_max = a_max                      # Set maximum a value to allow
        self.sigmaSQ = K.variable(value=0.01)   # Standard devation of normal distribution for error
        self.gammaSQ = K.variable(value=0.01)   # Scale of Cauchy distribution for error
        # Parameter Initialization - Conditional distribution of the latent variables
        if isinstance(x, list):
            self.T = np.zeros((x[0].shape[0], 2))
        else:
            self.T = np.zeros((self.x.shape[0], 2))
        self.T[:, 0] = np.random.uniform(size=self.T.shape[0])
        self.T[:, 1] = 1. - self.T[:, 0]
        self.T_k = K.variable(value=self.T)
        self.a = K.variable(value=np.mean(self.T[:, 0]))  # Probability of membership to normal distribution

        self.avalues = []  # array to store a evolution
        self.sigmaSQvalues = []  # array to store sigmaSQ evolution
        self.gammaSQvalues = []  # array to store gammaSQ evolution

    def on_epoch_end(self, epoch, logs={}):
        """ Updates the parameters of the distributions in the contamination model on epoch end. The parameters updated are: 'a' for the global weight of the membership to the normal distribution, 'sigmaSQ' for the variance of the normal distribution and 'gammaSQ' for the scale of the Cauchy distribution of outliers. The latent variables are updated as well: 'T_k' describing in the first column the probability of membership to normal distribution and in the second column probability of membership to the Cauchy distribution i.e. outlier. Stores evolution of global parameters (a, sigmaSQ and gammaSQ).

        Parameters
        ----------
        epoch : integer
            Current epoch in training.
        logs : keras logs
            Metrics stored during current keras training.
        """
        y_pred = self.model.predict(self.x)
        error = self.y.squeeze() - y_pred.squeeze()

        # Update parameters (M-Step)
        errorSQ = error**2
        aux = np.mean(self.T[:, 0])
        if aux > self.a_max:
            aux = self.a_max
        K.set_value(self.a, aux)
        K.set_value(self.sigmaSQ, np.sum(self.T[:, 0] * errorSQ) / np.sum(self.T[:, 0]))
        # Gradient descent
        gmSQ_eval = K.get_value(self.gammaSQ)
        grad_gmSQ = (0.5 * np.sum(self.T[:, 1]) - np.sum(self.T[:, 1] * errorSQ / (gmSQ_eval + errorSQ))) / gmSQ_eval
        # Guarantee positivity in update
        eta = K.get_value(self.model.optimizer.lr)
        new_gmSQ = gmSQ_eval - eta * grad_gmSQ
        while new_gmSQ < 0 or (new_gmSQ/gmSQ_eval) > 1000:
            eta /= 2
            new_gmSQ = gmSQ_eval - eta * grad_gmSQ
        K.set_value(self.gammaSQ, new_gmSQ)

        # Update conditional distribution of latent variables (beginning of E-Step)
        a_eval = K.get_value(self.a)
        sigmaSQ_eval = K.get_value(self.sigmaSQ)
        gammaSQ_eval = K.get_value(self.gammaSQ)
        print('a: %f, sigmaSQ: %f, gammaSQ: %f' % (a_eval, sigmaSQ_eval, gammaSQ_eval))
        norm_eval = norm.pdf(error, loc=0, scale=np.sqrt(sigmaSQ_eval))
        cauchy_eval = cauchy.pdf(error, loc=0, scale=np.sqrt(gammaSQ_eval))
        denominator = a_eval * norm_eval + (1. - a_eval) * cauchy_eval
        self.T[:, 0] = a_eval * norm_eval / denominator
        self.T[:, 1] = (1. - a_eval) * cauchy_eval / denominator
        K.set_value(self.T_k, self.T)

        # store evolution of global variables
        self.avalues.append(a_eval)
        self.sigmaSQvalues.append(sigmaSQ_eval)
        self.gammaSQvalues.append(gammaSQ_eval)


def mse_contamination_metric(nout):
    """This function computes the mean squared error (mse) for the contamination model. The mse is computed over the prediction. Therefore, the augmentation for the index variable is ignored.

    Parameters
    ----------
    nout : int
        Number of outputs without uq augmentation (in the contamination model the augmentation corresponds to the data index in training).
    """
    def metric(y_true, y_pred):
        """
        Parameters
        ----------
        y_true : Keras tensor
            Keras tensor including the ground truth. Since the keras tensor includes an extra column to store the index of the data sample in the training set this column is ignored.
        y_pred : Keras tensor
            Keras tensor with the predictions of the contamination model (no data index).
        """

        return mean_squared_error(y_true[:, :nout], y_pred[:, :nout])

    metric.__name__ = 'mse_contamination'
    return metric


def mae_contamination_metric(nout):
    """This function computes the mean absolute error (mae) for the contamination model. The mae is computed over the prediction. Therefore, the augmentation for the index variable is ignored.

    Parameters
    ----------
    nout : int
        Number of outputs without uq augmentation (in the contamination model the augmentation corresponds to the data index in training).
    """
    def metric(y_true, y_pred):
        """
        Parameters
        ----------
        y_true : Keras tensor
            Keras tensor including the ground truth. Since the keras tensor includes an extra column to store the index of the data sample in the training set this column is ignored.
        y_pred : Keras tensor
            Keras tensor with the predictions of the contamination model (no data index).
        """

        return mean_absolute_error(y_true[:, :nout], y_pred[:, :nout])

    metric.__name__ = 'mae_contamination'
    return metric


def r2_contamination_metric(nout):
    """This function computes the r2 for the contamination model. The r2 is computed over the prediction. Therefore, the augmentation for the index variable is ignored.

    Parameters
    ----------
    nout : int
        Number of outputs without uq augmentation (in the contamination model the augmentation corresponds to the data index in training).
    """
    def metric(y_true, y_pred):
        """
        Parameters
        ----------
        y_true : Keras tensor
            Keras tensor including the ground truth. Since the keras tensor includes an extra column to store the index of the data sample in the training set this column is ignored.
        y_pred : Keras tensor
            Keras tensor with the predictions of the contamination model (no data index).
        """
        #if nout > 1:
        #    y_true_ = K.reshape(y_true[:, :-1], K.shape(y_pred))
        #else:
        #    y_true_ = K.reshape(y_true[:, 0], K.shape(y_pred))
        y_true_ = K.reshape(y_true[:, :-1], K.shape(y_pred))

        SS_res = K.sum(K.square(y_true_ - y_pred))
        SS_tot = K.sum(K.square(y_true_ - K.mean(y_true_)))
        return (1. - SS_res/(SS_tot + K.epsilon()))

    metric.__name__ = 'r2_contamination'
    return metric
