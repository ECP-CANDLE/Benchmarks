import os
import sys
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from pytorch_pretrained_bert import BertTokenizer, BertConfig
import logging
import numpy as np
import pandas as pd
import random
import pickle
import sys


class MimicDatasetSynthetic(Dataset):
    """Synthetic dataset in the style of Mimic"""

    def __init__(
        self,
        size=60000,
        vocab_size=25000,
        n_classes=10,
        len_mean=4000,
        len_std=2000,
        max_seg=10,
    ):

        self.size = size
        self.n_classes = n_classes
        self.len_mean = len_mean
        self.len_std = len_std
        self.vocab_size = vocab_size
        self.max_seg = max_seg

        self.tokens = []
        self.masks = []
        self.seg_ids = []
        self.n_segs = []
        self.labels = []

        # generate random documents
        for i in range(self.size):
            # random document length, make sure not negative
            l = int(np.rint(np.random.normal(self.len_mean, self.len_std)))
            if l <= 0:
                l = 50

            # generate random tokens, avoid range of special tokens used for BERT tokenizer
            doc = np.random.randint(1000, self.vocab_size, l)[: max_seg * 512]

            # chunk into 512 length segments
            text_segments = [doc[j : j + 512] for j in range(0, l, 512)]

            tokens_ = []
            seg_ids_ = []
            masks_ = []
            self.n_segs.append(len(text_segments))

            # generate 10 random labels per document (multilabel task)
            one_hot = np.zeros((n_classes))
            label = np.random.randint(0, self.n_classes, 10)
            one_hot[label] = 1
            self.labels.append(one_hot)

            for j, text_segment in enumerate(text_segments):

                if len(text_segment) < 5:
                    continue

                text_segment = list(text_segment[:510])
                text_segment = (
                    [101] + text_segment + [102]
                )  # special start/end tokens used by BERT
                l = len(text_segment)
                l_pad = 512 - l
                text_segment += [0] * l_pad
                tokens_.append(text_segment)
                seg_ids_.append([0 for i in text_segment])
                masks_.append([1] * l + [0] * l_pad)

            # pad all batches to same size
            if len(tokens_) < max_seg:
                diff = self.max_seg - len(tokens_)
                tokens_ += [[0] * 512 for i in range(diff)]
                seg_ids_ += [[0] * 512 for i in range(diff)]
                masks_ += [[0] * 512 for i in range(diff)]

            self.tokens.append(tokens_)
            self.seg_ids.append(seg_ids_)
            self.masks.append(masks_)

            #sys.stdout.write("generating document %i      \r" % i)
            #sys.stdout.flush()

    def __len__(self):
        return self.size

    def __getitem__(self, idx):

        sample = {
            "tokens": torch.tensor(self.tokens[idx], dtype=torch.long),
            "masks": torch.tensor(self.masks[idx], dtype=torch.long),
            "seg_ids": torch.tensor(self.seg_ids[idx], dtype=torch.long),
            "n_segs": torch.tensor(self.n_segs[idx], dtype=torch.int),
            "label": torch.tensor(self.labels[idx], dtype=torch.float),
        }

        return sample
