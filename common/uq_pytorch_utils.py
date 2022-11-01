from __future__ import absolute_import

from typing import Tuple, Union

import torch
import torch.nn as nn


class abstention_loss(nn.Module):

    def __init__(self, alpha, mask):
        super(abstention_loss, self).__init__()
        self.alpha = alpha
        self.mask = mask
        self.ndevices = torch.cuda.device_count()
        self.eps = torch.finfo(torch.float32).eps

    def forward(self, y_pred, y_true):

        if self.ndevices > 0:
            loss_cross_entropy = nn.CrossEntropyLoss().cuda()
        else:
            loss_cross_entropy = nn.CrossEntropyLoss()

        base_pred = (1. - self.mask) * y_pred + self.eps
        base_true = y_true
        base_cost = loss_cross_entropy(base_pred, base_true)

        abs_pred = torch.sum(self.mask * y_pred, -1)
        # add some small value to prevent NaN when prediction is abstained
        abs_pred = torch.clamp(abs_pred, self.eps, 1. - self.eps)

        return ((1. - abs_pred) * base_cost - self.alpha * torch.log(1. - abs_pred))


class abstention_loss_ce(nn.Module):
    """ Function to compute abstention loss for classification problems.
        It is composed by two terms:
        (i) original loss of the multiclass classification problem using cross entropy,
        (ii) cost associated to the abstaining samples.
    """

    def __init__(self, alpha0: float,
                 alpha_scale_factor: float = 0.8,
                 min_abs_acc: float = 0.6,
                 max_abs_frac: float = 0.4,
                 acc_gain: float = 1.0,
                 abs_gain: float = 1.0,
                 ndevices: int = 0):
        """ Initialization of class for abstention loss using cross entropy.

        Parameters
        ----------
        alpha0 : Initial weight of abstention term in cost function
        alpha_scale_factor : Multiplier to scale weight of abstention term in cost function. Default: 0.8
        min_abs_acc : Minimum accuracy target with abstention. Default: 0.6
        max_abs_frac : Maximum abstention fraction target. Default: 0.4
        acc_gain : Multiplier for accuracy error (with respect to target) when updating alpha. Default: 1.0
        abs_gain : Multiplier for abstention fraction error (with respect to target) when updating alpha. Default: 1.0
        ndevices : Number of available GPUs.
        """
        super(abstention_loss_ce, self).__init__()
        self.ndevices = ndevices
        self.eps = torch.finfo(torch.float32).eps
        # weight for abstention term in cost function
        self.alpha = torch.autograd.Variable(torch.ones(1) * alpha0).float()
        # factor to scale alpha
        self.alpha_scale_factor = alpha_scale_factor
        # min target accuracy (value specified as parameter of the run)
        self.min_abs_acc = min_abs_acc
        # maximum abstention fraction (value specified
        # as parameter of the run)
        self.max_abs_frac = max_abs_frac
        # factor for adjusting alpha scale
        self.acc_gain = acc_gain
        # factor for adjusting alpha scale
        self.abs_gain = abs_gain
        # array to store alpha evolution
        self.alphavalues = []
        # store metrics
        self.abs_acc = []
        self.abs_frac = []

    def abstention_acc_metric(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """Metric of accuracy with abstention.

        Parameters
        ----------
        x : Prediction made by the model. It is assumed that this tensor includes extra columns to store the abstaining class.
        y : True values to predict
        """
        mask_noabs = x[:, -1].lt(0.5)
        ind_all = torch.arange(y.shape[0])
        base_pred = torch.index_select(x[:, :-1], 0, torch.masked_select(ind_all, mask_noabs))
        base_true = torch.index_select(y, 0, torch.masked_select(ind_all, mask_noabs))

        total = base_true.size(0)
        if total == 0:  # All abstention, then base accuracy
            total = y.size(0)
            _, cl_predicted = torch.max(x.data, 1)
            correct = (cl_predicted == y).sum().item()

            return correct / total

        _, cl_predicted = torch.max(base_pred.data, 1)
        correct = (cl_predicted == base_true).sum().item()

        return correct / total

    def abstention_frac_metric(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """Metric of abstention fraction.

        Parameters
        ----------
        x : Prediction made by the model. It is assumed that this tensor includes extra columns to store the abstaining class.
        y : True values to predict (not used)
        """
        xabs = x[:, -1].ge(0.5).float()
        abs_pred = xabs.mean()

        return abs_pred

    def update_alpha(self, abs_acc: float, abs_frac: float):
        """ This function adapts the parameter alpha in the abstention loss.

        The parameter alpha (weight of the abstention term in the abstention loss) is increased or decreased adaptively during the training run.
        It is decreased if the current abstention accuracy is less than the minimum target accuracy set or increased if the current abstention fraction is greater than the target maximum fraction set.
        Thresholds for minimum and maximum correction factors are computed and the correction over alpha is not allowed to be less or greater than them, respectively, to avoid huge swings in the abstention loss evolution.

        Parameters
        ----------
        abs_acc : Current accuracy taking abstention into account
        abs_frac : Current abstention fraction
        """
        # Current accuracy (with abstention)
        self.abs_acc.append(abs_acc)
        # Current abstention fraction
        self.abs_frac.append(abs_frac)

        # modify alpha as needed
        acc_error = self.min_abs_acc - abs_acc
        acc_error = max(acc_error, 0.0)
        abs_error = abs_frac - self.max_abs_frac
        abs_error = max(abs_error, 0.0)
        new_scale = 1.0 - self.acc_gain * acc_error + self.abs_gain * abs_error
        if new_scale < 0.:
            new_scale = 0.99

        # threshold to avoid huge swings
        min_scale = self.alpha_scale_factor
        max_scale = 1. / self.alpha_scale_factor
        new_scale = min(new_scale, max_scale)
        new_scale = max(new_scale, min_scale)

        print('Scaling factor: ', new_scale)
        self.alpha *= new_scale
        print('alpha: ', self.alpha)

        self.alphavalues.append(self.alpha.detach().numpy()[0])

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        """ Compute cross entropy abstention loss.

        Parameters
        ----------
        x : Prediction made by the model.
            It is assumed that this tensor includes extra columns to store the abstaining classes.
        y : True values to predict
        """
        xabs = nn.Sigmoid(x[:, -1] - 0.5)

        # add some small value to prevent NaN when prediction is abstained
        xabs = torch.clamp(xabs, min=self.eps, max=1. - self.eps)

        # Cross Entropy
        if self.ndevices > 0:
            loss_ce = nn.CrossEntropyLoss(reduction='none').cuda()
        else:
            loss_ce = nn.CrossEntropyLoss(reduction='none')

        base_cost = loss_ce(x[:, :-1], y)

        # Average over all the samples
        return torch.mean((1. - xabs) * base_cost - self.alpha * torch.log(1. - xabs))


class abstention_loss_mse(nn.Module):
    """ Function to compute abstention loss for regression problems.
        It is composed by two terms:
        (i) original loss of the regression problem using mean squared error,
        (ii) cost associated to the abstaining samples.
    """

    def __init__(self, alpha0: float,
                 alpha_scale_factor: float = 0.8,
                 max_abs_loss: float = 1.0,
                 max_abs_frac: float = 0.4,
                 loss_gain: Union[float, Tuple[float]] = 1.0,
                 abs_gain: Union[float, Tuple[float]] = 1.0,
                 ndevices: int = 0):
        """ Initialization of class for abstention loss using mean squared error.

        Parameters
        ----------
        alpha0 : Initial weight of abstention term in cost function
        alpha_scale_factor : Multiplier to scale weight of abstention term in cost function. Default: 0.8
        max_abs_loss : Maximum loss target with abstention. Default: 1.0
        max_abs_frac : Maximum abstention fraction target. Default: 0.4
        loss_gain : Multiplier for loss error (with respect to target) when updating alpha. If a tuple is provided a PID-based adaptation is used. Default: 1.0
        abs_gain : Multiplier for abstention fraction error (with respect to target) when updating alpha. If a tuple is provided a PID-based adaptation is used. Default: 1.0
        ndevices : Number of available GPUs.
        """
        super(abstention_loss_mse, self).__init__()
        self.ndevices = ndevices
        self.eps = torch.finfo(torch.float32).eps
        # weight for abstention term in cost function
        self.alpha = torch.autograd.Variable(torch.ones(1) * alpha0).float()
        # factor to scale alpha
        self.alpha_scale_factor = alpha_scale_factor
        # max target loss (value specified as parameter of the run)
        self.max_abs_loss = max_abs_loss
        # maximum abstention fraction (value specified
        # as parameter of the run)
        self.max_abs_frac = max_abs_frac
        # factor for adjusting alpha scale
        self.loss_gain = loss_gain
        # factor for adjusting alpha scale
        self.abs_gain = abs_gain
        # array to store alpha evolution
        self.alphavalues = []
        # store metrics
        self.abs_loss = []
        self.abs_frac = []

        # previous abs_loss (for PID-based adaptation)
        self.abs_loss_old = None
        # previous abs_frac (for PID-based adaptation)
        self.abs_frac_old = None
        # multipliers for integral term (for PID-based adaptation)
        self.loss_int = 0.0
        self.abs_int = 0.0

        self.alpha_adapt_func = self.update_alpha
        if isinstance(loss_gain, tuple) or isinstance(abs_gain, tuple):
            # PID-based adaptation
            self.alpha_adapt_func = self.update_alpha_pid

    def abstention_loss_metric(self, x: torch.Tensor, y: torch.Tensor):
        """Metric of MSE-based loss with abstention. The loss is normalized to have a fixed scale for the target loss.

        Parameters
        ----------
        x : Prediction made by the model. It is assumed that this tensor includes an extra column to store the abstaining indicator.
        y : True values to predict
        """
        mask_noabs = x[:, -1].lt(0.5)

        ind_all = torch.arange(y.shape[0])
        base_pred = torch.index_select(x[:, 0], 0, torch.masked_select(ind_all, mask_noabs))
        base_true = torch.index_select(y, 0, torch.masked_select(ind_all, mask_noabs))

        max_true = y.max()
        base_true_n = base_true / max_true
        base_pred_n = base_pred / max_true

        return torch.nn.functional.mse_loss(base_pred_n, base_true_n)

    def abstention_frac_metric(self, x: torch.Tensor, y: torch.Tensor):
        """Metric of abstention fraction.

        Parameters
        ----------
        x : Prediction made by the model. It is assumed that this tensor includes extra columns to store the abstaining class.
        y : True values to predict (not used)
        """
        xabs = x[:, -1].ge(0.5).float()
        abs_pred = xabs.mean()

        return abs_pred

    def update_alpha(self, abs_loss: float, abs_frac: float):
        """ This function adapts the parameter alpha in the abstention loss.

        The parameter alpha (weight of the abstention term in the abstention loss) is increased or decreased adaptively during the training run.
        It is decreased if the current abstention loss is greater than the maximum target loss set or increased if the current abstention fraction is greater than the maximum target fraction set.
        Thresholds for minimum and maximum correction factors are computed and the correction over alpha is not allowed to be less or greater than them, respectively, to avoid huge swings in the abstention loss evolution.

        Parameters
        ----------
        abs_loss : Current (normalized) loss taking abstention into account
        abs_frac : Current abstention fraction
        """
        # Current loss (with abstention)
        self.abs_loss.append(abs_loss)
        # Current abstention fraction
        self.abs_frac.append(abs_frac)

        # modify alpha as needed
        loss_error = abs_loss - self.max_abs_loss
        loss_error = max(loss_error, 0.0)
        abs_error = abs_frac - self.max_abs_frac
        abs_error = max(abs_error, 0.0)
        new_scale = 1.0 - self.loss_gain * loss_error + self.abs_gain * abs_error
        if new_scale < 0.:
            new_scale = 0.99
        # threshold to avoid huge swings
        min_scale = self.alpha_scale_factor
        max_scale = 1. / self.alpha_scale_factor
        new_scale = min(new_scale, max_scale)
        new_scale = max(new_scale, min_scale)

        print('Scaling factor: ', new_scale)
        self.alpha *= new_scale
        print('alpha: ', self.alpha)

        self.alphavalues.append(self.alpha.detach().numpy())

    def update_alpha_pid(self, abs_loss: float, abs_frac: float):
        """ This function adapts the parameter alpha in the abstention loss using a PID-based adaptation.

        The parameter alpha (weight of the abstention term in the abstention loss) is increased or decreased adaptively during the training run.
        It is decreased if the current abstention loss is greater than the maximum target loss set or increased if the current abstention fraction is greater than the maximum target fraction set. PID factors for each of these criteria are used to stabilize the adaptation.
        Thresholds for minimum and maximum correction factors are computed and the correction over alpha is not allowed to be less or greater than them, respectively, to avoid huge swings in the abstention loss evolution.

        Parameters
        ----------
        abs_loss : Current (normalized) loss taking abstention into account
        abs_frac : Current abstention fraction
        """
        # Current loss (with abstention)
        self.abs_loss.append(abs_loss)
        # Current abstention fraction
        self.abs_frac.append(abs_frac)

        # modify alpha as needed
        # proportional term
        loss_error = abs_loss - self.max_abs_loss
        loss_error = max(loss_error, 0.0)
        abs_error = abs_frac - self.max_abs_frac
        abs_error = max(abs_error, 0.0)
        p_term = - self.loss_gain[0] * loss_error + self.abs_gain[0] * abs_error
        # integral term
        self.loss_int = self.loss_int + abs_loss
        self.abs_int = self.abs_int + abs_frac
        i_term = - self.loss_gain[1] * self.loss_int + self.abs_gain[1] * self.abs_int
        # derivative term
        if self.abs_loss_old is not None:
            dle = abs_loss - self.abs_loss_old
            dla = abs_frac - self.abs_frac_old
        else:
            dle = 0.0
            dla = 0.0
        self.abs_loss_old = abs_loss
        self.abs_frac_old = abs_frac
        d_term = - self.loss_gain[2] * dle + self.abs_gain[2] * dla

        new_scale = 1.0 + p_term + i_term + d_term
        if new_scale < 0.:
            new_scale = 0.99
        # threshold to avoid huge swings
        min_scale = self.alpha_scale_factor
        max_scale = 1. / self.alpha_scale_factor
        new_scale = min(new_scale, max_scale)
        new_scale = max(new_scale, min_scale)

        print('Scaling factor: ', new_scale)
        self.alpha *= new_scale
        self.alphavalues.append(self.alpha.detach().numpy())
        print('alpha: ', self.alphavalues[-1])

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        """ Compute mean squared error abstention loss.

        Parameters
        ----------
        x : Prediction made by the model.
            It is assumed that this tensor includes an extra column to store the abstaining class.
        y : True values to predict
        """
        xabs = nn.Sigmoid(x[:, -1] - 0.5)

        # add some small value to prevent NaN when prediction is abstained
        xabs = torch.clamp(xabs, min=self.eps, max=1. - self.eps)

        # Squared Error
        base_cost = torch.sum((x[:, :-1] - y)**2, dim=-1)

        # Average over all the samples
        return torch.mean((1. - xabs) * base_cost - self.alpha * torch.log(1. - xabs))
