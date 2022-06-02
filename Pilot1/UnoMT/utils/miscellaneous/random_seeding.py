"""
    File Name:          UnoPytorch/random_seeding.py
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               8/22/18
    Python Version:     3.6.6
    File Description:

"""
import random
import numpy as np
import torch


def seed_random_state(rand_state: int = 0):
    """seed_random_state(0)

    This function sets up with random seed in multiple libraries possibly used
    during PyTorch training and validation.

    Args:
        rand_state (int): random seed

    Returns:
        None
    """

    random.seed(rand_state)

    np.random.seed(rand_state)

    torch.manual_seed(rand_state)
    torch.cuda.manual_seed_all(rand_state)

    # This must be set to True for deterministic results in PyTorch 4.1
    torch.backends.cudnn.deterministic = True
