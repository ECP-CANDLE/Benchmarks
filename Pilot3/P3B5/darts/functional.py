import torch


def flatten(tensor):
    """ Flatten a tensor.

    Parameters
    ----------
    tensor : torch.tensor

    Returns
    -------
    Flattened tensor

    Example
    -------
    >>> x = torch.tensor([[0,1],[2,3]])
    >>> x_flattened = flatten(x)
    >>> print(x)
    >>> tensor([[0, 1],
                [2, 3]])
    >>> print(x_flattened)
    >>> tensor([0, 1, 2, 3])
    """
    return torch.cat([x.view(-1) for x in tensor])


def multitask_loss(logits, target, criterion, reduce='mean'):
    """ Compute multitask loss """
    losses = {}
    for task, label in target.items():
        losses[task] = criterion(logits[task], label)

    if reduce:
        total = 0
        for _, value in losses.items():
            total += value

        if reduce == "mean":
            losses = total / len(losses)
        elif reduce == "sum":
            losses = total
        else:
            raise ValueError('Reduced loss must use either `mean` or `sum`!')

    return losses