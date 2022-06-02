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


def multitask_loss(target, logits, criterion, reduce='mean'):
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


def accuracy(target: torch.tensor, output: torch.tensor,):
    """ Computes accuracy

    Args:
        output: logits of the model
        target: true labels

    Returns:
        accuracy of the predictions
    """
    return output.argmax(1).eq(target).double().mean().item()


def multitask_accuracy(target, output):
    """ Compute the accuracy for multitask problems """
    accuracies = {}
    for key, value in target.items():
        accuracies[key] = accuracy(target[key], output[key])

    return accuracies


def accuracy_topk(target, output, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def multitask_accuracy_topk(target, output, topk=(1,)):
    """Compute the topk accuracy for multitask problems"""
    topk_accuracies = {}
    for key, value in target.items():
        topk_accuracies[key] = accuracy_topk(output[key], target[key], topk)

    return topk_accuracies
