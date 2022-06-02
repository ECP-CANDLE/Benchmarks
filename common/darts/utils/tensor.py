import torch


def to_device(tensor, device: torch.device):
    """ Convert tensor-like object to given PyTorch device """
    if tensor is None:
        return tensor
    elif isinstance(tensor, torch.Tensor):
        return tensor.to(device)
    elif isinstance(tensor, dict):
        return {k: to_device(v, device) for k, v in tensor.items()}
    elif isinstance(tensor, list):
        return [to_device(v, device) for v in tensor]
    elif isinstance(tensor, tuple):
        return tuple(to_device(v, device) for v in tensor)
    else:
        raise NotImplementedError
