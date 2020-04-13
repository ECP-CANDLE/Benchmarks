from __future__ import absolute_import


from keras import backend as K

from keras.callbacks import Callback

from keras.models import Model
from keras.layers import Dense

from keras.utils import np_utils

import numpy as np

###################################################################

# For Abstention Model

# These are the parameters of the abstention loss
mu = None   # Factor weighting abstention term in cost function (auto tunes)
mask = None # Mask for abstention: it is 1 on the output associated to the
            # abstention class and 0 otherwise
nb_classes = None # integer or vector of integers with the index of the abstention class

def abstention_variable_initialization(mu0, mask_, nb_classes_):
    """ Function that initializes parameters of the abstention loss
    
    Parameters
    ----------
    mu0 : float
        Initial weight of abstention term in cost function
    mask_ : ndarray
        Numpy array to use as initialiser for global mask variable
    nb_classes_ : int or ndarray
        Integer or numpy array defining indices of the abstention class
    """

    global mu, mask, nb_classes

    # Parameter Initialization
    mu = K.variable(value=mu0)     # Weight of abstention term
    mask = K.variable(value=mask_) # Mask to compute cost
    nb_classes = K.variable(value=nb_classes_, dtype='int64') # integer or vector of integers with the index of the abstention class


def abstention_loss(y_true, y_pred):
    """ Function to compute abstention loss. It is composed by two terms: (i) original loss of the multiclass classification problem, (ii) cost associated to the abstaining samples.
    
    Parameters
    ----------
    y_true : keras tensor
        True values to predict
    y_pred : keras tensor
        Prediction made by the model. It is assumed that this keras tensor includes extra columns to store the abstaining classes.
    """
    cost = 0
    base_pred = (1-mask)*y_pred
    base_true = y_true
    base_cost = K.categorical_crossentropy(base_true,base_pred)
    abs_pred = K.mean(mask*y_pred, axis=-1)
    cost = (1.-abs_pred)*base_cost - mu*K.log(1.-abs_pred)

    return cost


def abs_acc(y_true, y_pred):
    """ Function to estimate accuracy over the predicted samples after removing the samples where the model is abstaining.
    
    Parameters
    ----------
    y_true : keras tensor
        True values to predict
    y_pred : keras tensor
        Prediction made by the model. It is assumed that this keras tensor includes extra columns to store the abstaining classes.
    """

    # matching in original classes
    true_pred = K.sum(K.cast(K.equal(K.argmax(y_true, axis=-1), K.argmax(y_pred, axis=-1)), 'int64'))

    # total abstention
    total_abs = K.sum(K.cast(K.equal(K.argmax(y_pred, axis=-1), nb_classes), 'int64'))

    # total predicted in original classes
    total_pred = K.sum(K.cast(K.equal(K.argmax(y_pred, axis=-1), K.argmax(y_pred, axis=-1)), 'int64'))

    return true_pred/(total_pred - total_abs)


def acc_class1(y_true, y_pred):
    """ Function to estimate accuracy over the class 1 prediction. This estimation is global (i.e. abstaining samples are not removed)
    
    Parameters
    ----------
    y_true : keras tensor
        True values to predict
    y_pred : keras tensor
        Prediction made by the model. It is assumed that this keras tensor includes extra columns to store the abstaining classes.
    """

    # Find samples in ground truth belonging to class 1
    ytrueint = K.argmax(y_true, axis=-1)

    # Compute total number of ground truth samples in class 1
    total_true1 = K.sum(ytrueint)

    # Find samples in prediction belonging to class 1
    ypredint = K.argmax(y_pred[:,:2], axis=-1)

    # Find correctly predicted class 1 samples
    true1_pred = K.sum(ytrueint*ypredint)

    # Compute accuracy in class 1
    acc = true1_pred / total_true1

    # Since there are so few samples in class 1
    # it is possible that ground truth does not
    # have any sample in class 1, leading to a divide
    # by zero and not valid accuracy
    # Therefore, for the accuracy to be valid
    # total_true1 should be greater than zero
    # otherwise, return 0.

    condition = K.greater(total_true1, 0)

    return K.switch(condition, acc, K.zeros_like(acc, dtype=acc.dtype))


def abs_acc_class1(y_true, y_pred):
    """ Function to estimate accuracy over the class 1 prediction after removing the samples where the model is abstaining
    
    Parameters
    ----------
    y_true : keras tensor
        True values to predict
    y_pred : keras tensor
        Prediction made by the model. It is assumed that this keras tensor includes extra columns to store the abstaining classes.
    """

    # Find locations of true 1 prediction
    ytrueint = K.argmax(y_true, axis=-1)

    # Find locations that are predicted (not abstained)
    mask_pred = K.cast(K.not_equal(K.argmax(y_pred, axis=-1), nb_classes), 'int64')

    # Compute total number of ground truth samples in class 1 filtering abstaining predictions
    total_true1 = K.sum(ytrueint * mask_pred)

    # matching in original class 1 after removing abstention
    true1_pred = K.sum(mask_pred * ytrueint * K.cast(K.equal(K.argmax(y_true, axis=-1), K.argmax(y_pred, axis=-1)), 'int64'))

    # Compute accuracy in class 1
    acc = true1_pred / total_true1

    # Since there are so few samples in class 1
    # it is possible that ground truth does not
    # have any sample in class 1, leading to a divide
    # by zero and not valid accuracy
    # Therefore, for the accuracy to be valid
    # total_true1 should be greater than zero
    # otherwise, return 0.

    condition = K.greater(total_true1, 0)

    return K.switch(condition, acc, K.zeros_like(acc, dtype=acc.dtype))


class AbstentionAdapt_Callback(Callback):
    """ This callback is used to adapt the parameter mu in the abstention loss.
        Factor mu (weight of the abstention term in the abstention loss) is increased or decreased to try to match the accuracy set as target.
        The accuracy to use must be specified as the 'monitor' argument in the initialization of the callback. It could be: the accuracy without abstention samples (abs_acc), the accuracy over class 1 without abstention samples (abs_acc_class1), etc.
        If the current monitored accuracy is smaller than the target set, mu increases to promote more abstention.
        If the current monitored accuracy is greater than the target set, mu decreases to promote more predictions (less abstention).
    """

    def __init__(self, monitor, init_abs_epoch=4, scale_factor=0.95, target_acc=0.95):
        """ Initializer of the AbstentionAdapt_Callback.
        Parameters
        ----------
        monitor : keras metric
            Metric to monitor during the run and use as base to adapt the weight of the abstention term (i.e. mu) in the asbstention cost function
        init_abs_epoch : integer
            Value of the epochs to start adjusting the weight of the abstention term (i.e. mu). Default: 4.
        scale_factor: float
            Factor to scale (increase by dividing or decrease by multiplying) the weight of the abstention term (i.e. mu). Default: 0.95.
        target_acc: float
            Target accuracy to achieve in the current training. Default: 0.95.
        """
        super(AbstentionAdapt_Callback, self).__init__()

        self.monitor = monitor
        self.init_abs_epoch = init_abs_epoch # epoch to init abstention
        self.scale_factor = scale_factor # factor to scale mu (weight for abstention term in cost function)
        self.target_acc = target_acc # target accuracy (value specified as parameter of the run)
        self.muvalues = [] # array to store mu evolution

        
    def on_epoch_end(self, epoch, logs=None):
        """ Initializer of the AbstentionAdapt_Callback.
        Parameters
        ----------
        epoch : integer
            Current epoch in training.
        logs : keras logs
            Metrics stored during current keras training.
        """

        new_mu_val = K.get_value(mu)
        if epoch > self.init_abs_epoch:
        
            current = logs.get(self.monitor)
            
            if current is None:
                warnings.warn( 'Abstention Adapt conditioned on metric `%s` ' 'which is not available. Available metrics are: %s' % (self.monitor, ','.join(list(logs.keys()))), RuntimeWarning)
            else:
                # modify mu as needed
                if current > self.target_acc: #increase abstention penalty
                    new_mu_val /= self.scale_factor
                elif current < self.target_acc: #decrease abstention penalty
                    new_mu_val *= self.scale_factor

                K.set_value(mu, new_mu_val)
        self.muvalues.append( new_mu_val )

        #print('epoch: %d, mu: %f' % (epoch, new_mu_val))


def modify_labels(numclasses_out, ytrain, ytest, yval):
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
    classesval = np.max(yval) + 1

    assert( classestrain == classestest )
    assert( classesval == classestest )
    assert( (classestrain+1) == numclasses_out ) # In this case only one other slot for abstention is created

    labels_train = np_utils.to_categorical( ytrain, numclasses_out )
    labels_test = np_utils.to_categorical( ytest, numclasses_out )
    labels_val = np_utils.to_categorical( yval, numclasses_out )

    # For sanity check
    mask_vec = np.zeros(labels_train.shape)
    mask_vec[:,-1] = 1
    i = np.random.choice(range(labels_train.shape[0]))
    sanity_check = mask_vec[i,:]*labels_train[i,:]
    print(sanity_check.shape)
    if ytrain.ndim > 1:
        ll = ytrain.shape[1]
    else:
        ll = 0
        
    for i in range( ll ):
        for j in range( numclasses_out ):
            if sanity_check[i,j] == 1:
                print('Problem at ',i,j)

    return labels_train, labels_test, labels_val

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
        'het' for heteroscedastic regression which needs the outputs to be doubled. (current implicit default: 'het')
    num_add : integer
        Number of units to add. This only applies to the 'abstain' mode.
    activation : string
        String with keras specification of activation function (e.g. 'relu', 'sigomid', 'softmax', etc.)
        
    Return
    ----------
    modelOut : keras model
        Keras model after last dense layer has been modified as specified. If there is no mode specified it returns the same model.
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
    if mode is 'abstain':
        assert num_add is not None
        new_output_size = modelIn.layers[i].output_shape[-1] + num_add
    elif mode is 'qtl': # for quantile UQ
        new_output_size = 3 * modelIn.layers[i].output_shape[-1]
    else: # for heteroscedastic UQ
        new_output_size = 2 * modelIn.layers[i].output_shape[-1]

    # Recover current layer options
    config = modelIn.layers[i].get_config()
    # Update number of units
    config['units'] = new_output_size
    # Update activation function if requested
    if activation is not None:
        config['activation'] = activation
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
