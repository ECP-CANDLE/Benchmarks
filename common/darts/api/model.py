import hashlib

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """ Class representing sampleable neural network model """

    def num_params(self):
        """ Get the number of model parameters. """
        return sum(p.numel() for p in self.parameters())

    def summary(self, hashsummary=False):
        print(self)
        print("-" * 80)
        n_params = self.num_params()
        print(f"Number of model parameters: {n_params}")
        print("-" * 80)

        if hashsummary:
            print('Hash Summary:')
            for idx, hashvalue in enumerate(self.hashsummary()):
                print(f"{idx}: {hashvalue}")

    def hashsummary(self):
        """ Print a model summary - checksums of each layer parameters """
        children = list(self.children())

        result = []
        for child in children:
            result.extend(hashlib.sha256(x.detach().cpu().numpy().tobytes()).hexdigest() for x in child.parameters())

        return result

    def loss(self, x_data, y_true, reduce='mean'):
        """ Forward propagate network and return a value of loss function """
        # TODO: This may need to be moved to the model.
        if reduce not in (None, 'sum', 'mean'):
            raise ValueError("`reduce` must be either None, `sum`, or `mean`!")

        y_pred = self(x_data)
        return y_pred, self.loss_value(x_data, y_true, y_pred, reduce=reduce)

    def loss_value(self, x_data, y_true, y_pred, reduce=None):
        """ Calculate a value of loss function """
        raise NotImplementedError
