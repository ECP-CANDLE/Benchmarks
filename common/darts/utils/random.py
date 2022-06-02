import os
import torch
import random
import numpy as np


class Seeds:
    pythonhash = 0
    pythonrand = 0
    numpy = 0
    torch = 0


class SeedControl:

    def __init__(self, seeds=Seeds()):
        self.s = seeds

    def fix_all_seeds(self, seed: int):
        """Fix all seeds to the same seed"""
        self.s.pythonhash = seed
        self.s.pythonrand = seed
        self.s.numpy = seed
        self.s.torch = seed
        self.set_seeds()

    def set_seeds(self):
        os.environ['PYTHONHASHSEED'] = str(self.s.pythonhash)
        random.seed(self.s.pythonrand)
        np.random.seed(self.s.numpy)
        torch.random.manual_seed(self.s.torch)

    def get_seeds(self):
        return {
            'PythonHash': self.s.pythonhash,
            'PythonRand': self.s.pythonrand,
            'Numpy': self.s.numpy,
            'Torch': self.s.torch
        }
