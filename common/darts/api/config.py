import os
import datetime as dtm
from collections import namedtuple

import torch


def banner(device):
    """ Print a banner of the system config

    Parameters
    ----------
    device : torch.device
    """
    print("=" * 80)
    info = get_torch_info()
    torch_msg = (
        f"Pytorch version: {info.torch_version} ",
        f"cuda version {info.cuda_version} ",
        f"cudnn version {info.cudnn_version}"
    )
    print(''.join(torch_msg))

    if device.type == 'cuda':
        device_idx = get_device_idx(device)
        usage = memory_usage(device)
        print(f"CUDA Device name {torch.cuda.get_device_name(device_idx)}")
        print(f"CUDA memory - total: {usage.total} current usage: {usage.used}")
    else:
        print(f'Using CPU')

    print(dtm.datetime.now().strftime("%Y/%m/%d - %H:%M:%S"))
    print("=" * 80)


def get_torch_info():
    """ Get Pytorch system info """
    VersionInfo = namedtuple(
        "PytorchVersionInfo",
        "torch_version cuda_version cudnn_version"
    )
    return VersionInfo(torch.__version__, torch.version.cuda, torch.backends.cudnn.version())


def get_device_idx(device):
    """ Get the CUDA device from torch

    Parameters
    ----------
    device : torch.device

    Returns
    -------
    index of the CUDA device
    """
    return 0 if device.index is None else device.index


def memory_usage(device):
    """ Get GPU memory total and usage

    Parameters
    ----------
    device : torch.device

    Returns
    -------
    usage : namedtuple(torch.device, int, int)
        Total memory of the GPU and its current usage
    """
    if device.type == "cpu":
        raise ValueError(f'Can only query GPU memory usage, but device is {device}')

    Usage = namedtuple("MemoryUsage", "device total used")

    if device.type == "cuda":
        device_idx = get_device_idx(device)

    try:
        total, used = os.popen(
            'nvidia-smi --query-gpu=memory.total,memory.used --format=csv,nounits,noheader'
        ).read().split('\n')[device_idx].split(',')
    except ValueError:
        print(f'Attempted to query CUDA device {device_idx}, does this system have that many GPUs?')

    return Usage(device, int(total), int(used))
