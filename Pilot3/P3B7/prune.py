import torch
import numpy

import torch.nn as nn
import torch.nn.utils.prune as prune

from torch.nn.utils.prune import (
    BasePruningMethod, _validate_pruning_amount_init,
    _compute_nparams_toprune, _validate_pruning_amount
)


class MinMaxPrune(BasePruningMethod):

    PRUNING_TYPE = "unstructured"

    def __init__(self, amount):
        # Check range of validity of pruning amount
        _validate_pruning_amount_init(amount)
        self.amount = amount

    def compute_mask(self, t, default_mask):
        # Check that the amount of units to prune is not > than the number of
        # parameters in t
        tensor_size = t.nelement()
        tNP = t.detach().cpu().numpy()
        # Compute number of units to prune: amount if int,
        # else amount * tensor_size
        nparams_toprune = _compute_nparams_toprune(self.amount, tensor_size)
        # This should raise an error if the number of units to prune is larger
        # than the number of units in the tensor
        _validate_pruning_amount(nparams_toprune, tensor_size)

        mask = default_mask.clone(memory_format=torch.contiguous_format)

        if nparams_toprune != 0:  # k=0 not supported by torch.kthvalue
            # largest=True --> top k; largest=False --> bottom k
            # Prune the smallest k
            # topk = torch.topk(
            #     torch.abs(t).view(-1), k=nparams_toprune, largest=False
            # )
            # # topk will have .indices and .values
            # mask.view(-1)[topk.indices] = 0
            values = []
            indices = []
            for i in range(0, nparams_toprune):
                best = self.minimax(0, 0, True, tNP, -1000, 1000)
                numpy.append(values, best)
                bestInd = numpy.where(tNP == best)
                numpy.append(indices, bestInd)
                tNP = numpy.delete(tNP, bestInd)
            mask.view(-1)[indices] = 0
        return mask

    def minimax(self, depth, nodeIndex, maximizingPlayer, values, alpha, beta):
        # https://www.geeksforgeeks.org/minimax-algorithm-in-game-theory-set-4-alpha-beta-pruning/
        MIN = -1000
        MAX = 1000
        # Terminating condition. i.e
        # leaf node is reached
        if depth == 3:
            return values[nodeIndex]

        if maximizingPlayer:

            best = MIN

            # Recur for left and right children
            for i in range(0, 2):

                val = self.minimax(depth + 1, nodeIndex * 2 + i,
                                   False, values, alpha, beta)
                best = max(best, val)
                alpha = max(alpha, best)

                # Alpha Beta Pruning
                if beta <= alpha:
                    break

            return best

        else:
            best = MAX

            # Recur for left and
            # right children
            for i in range(0, 2):

                val = self.minimax(depth + 1, nodeIndex * 2 + i,
                                   True, values, alpha, beta)
                best = min(best, val)
                beta = min(beta, best)

                # Alpha Beta Pruning
                if beta <= alpha:
                    break

            return best

    @classmethod
    def apply(cls, module, name, amount):
        r"""Adds the forward pre-hook that enables pruning on the fly and
        the reparametrization of a tensor in terms of the original tensor
        and the pruning mask.

        Args:
            module (nn.Module): module containing the tensor to prune
            name (str): parameter name within ``module`` on which pruning
                will act.
            amount (int or float): quantity of parameters to prune.
                If ``float``, should be between 0.0 and 1.0 and represent the
                fraction of parameters to prune. If ``int``, it represents the
                absolute number of parameters to prune.
        """
        return super(MinMaxPrune, cls).apply(module, name, amount=amount)


class NegativePrune(BasePruningMethod):

    PRUNING_TYPE = "unstructured"

    def __init__(self, amount):
        # Check range of validity of pruning amount
        _validate_pruning_amount_init(amount)
        self.amount = amount

    def compute_mask(self, t, default_mask):
        # Check that the amount of units to prune is not > than the number of
        # parameters in t
        tensor_size = t.nelement()
        tNP = t.detach().cpu().numpy()
        # Compute number of units to prune: amount if int,
        # else amount * tensor_size
        nparams_toprune = _compute_nparams_toprune(self.amount, tensor_size)
        # This should raise an error if the number of units to prune is larger
        # than the number of units in the tensor
        _validate_pruning_amount(nparams_toprune, tensor_size)

        mask = default_mask.clone(memory_format=torch.contiguous_format)

        if nparams_toprune != 0:  # k=0 not supported by torch.kthvalue
            # largest=True --> top k; largest=False --> bottom k
            # Prune the smallest k
            # topk = torch.topk(
            #     torch.abs(t).view(-1), k=nparams_toprune, largest=False
            # )
            # topk will have .indices and .values
            t[t < 0] = 0
            indices = torch.nonzero((t == 0), as_tuple=True)
            mask.view(-1)[indices] = 0
        return mask

    @classmethod
    def apply(cls, module, name, amount):
        r"""Adds the forward pre-hook that enables pruning on the fly and
        the reparametrization of a tensor in terms of the original tensor
        and the pruning mask.

        Args:
            module (nn.Module): module containing the tensor to prune
            name (str): parameter name within ``module`` on which pruning
                will act.
            amount (int or float): quantity of parameters to prune.
                If ``float``, should be between 0.0 and 1.0 and represent the
                fraction of parameters to prune. If ``int``, it represents the
                absolute number of parameters to prune.
        """
        return super(NegativePrune, cls).apply(module, name, amount=amount)


def min_max_prune(module, name, amount):
    """Prune tensor according to min max game theory"""
    MinMaxPrune.apply(module, name, amount)
    return module


def negative_prune(module, name, amount):
    """Prune negative tensors"""
    NegativePrune.apply(module, name, amount)
    return module


def get_layers_to_prune(model: nn.Module):
    """Get layers to be pruned"""
    layers = []
    for name, module in model.named_modules():
        # prune amount % of connections in all 1D-conv layers
        if isinstance(module, torch.nn.Conv1d):
            layers.append((module, 'weight'))
        # prune amount/2 of connections in all linear layers
        elif isinstance(module, torch.nn.Linear):
            layers.append((module, 'weight'))
            print(f'Pruning {module}')

    return layers


def create_prune_masks(model: nn.Module):
    """Update the `model` with pruning masks.

    Args:
        model: model to be pruned

    Returns:
        model: model with pruning masks
    """
    for name, module in model.named_modules():
        # prune 40% of connections in all 1D-conv layers
        if isinstance(module, torch.nn.Conv1d):
            prune.l1_unstructured(module, name='weight', amount=0.4)
        # prune 20% of connections in all linear layers
        elif isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=0.2)

    return model


def remove_prune_masks(model: nn.Module):
    """Remove the `model` pruning masks.

    This is called after training with pruning so that
    we can save our model weights without the
    reparametrization of pruning.

    Args:
        model: model to be pruned

    Returns:
        model: model with pruning masks
    """
    for name, module in model.named_modules():
        # prune 40% of connections in all 1D-conv layers
        if isinstance(module, torch.nn.Conv1d):
            prune.remove(module, name='weight')
        # prune 20% of connections in all linear layers
        elif isinstance(module, torch.nn.Linear):
            prune.remove(module, name='weight', amount=0.2)

    return model
