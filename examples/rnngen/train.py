import argparse
from model.vocab import get_vocab_from_file, START_CHAR, END_CHAR
from model.model import CharRNN
import torch.utils.data
import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.rnn
import torch.nn.functional as F
from tqdm import tqdm
import os


def getconfig(args):
    return args


def count_valid_samples(smiles):
    from rdkit import Chem
    count = 0
    for smi in smiles:
        try:
            mol = Chem.MolFromSmiles(smi[1:-1])
        except Exception:
            continue
        if mol is not None:
            count += 1
    return count
