import torch

import torch.nn as nn
import torch.nn.utils.prune as prune


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
