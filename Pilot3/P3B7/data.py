import os
import pickle
import shutil
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class P3B3(Dataset):
    """P3B3 Synthetic Dataset.

    Args:
        root: str
            Root directory of dataset where CANDLE loads P3B3 data.

        partition: str
            dataset partition to be loaded.
            Must be either 'train' or 'test'.
    """

    training_data_file = "train_X.npy"
    training_label_file = "train_Y.npy"
    test_data_file = "test_X.npy"
    test_label_file = "test_Y.npy"

    def __init__(
        self,
        root,
        partition,
        subsite=True,
        laterality=True,
        behavior=True,
        grade=True,
        transform=None,
        target_transform=None,
    ):
        self.root = root
        self.partition = partition
        self.transform = transform
        self.target_transform = target_transform
        self.subsite = subsite
        self.laterality = laterality
        self.behavior = behavior
        self.grade = grade

        if self.partition == "train":
            data_file = self.training_data_file
            label_file = self.training_label_file
        elif self.partition == "test":
            data_file = self.test_data_file
            label_file = self.test_label_file
        else:
            raise ValueError("Partition must either be 'train' or 'test'.")

        self.data = np.load(os.path.join(self.root, data_file))
        self.targets = self.get_targets(label_file)

    def __repr__(self):
        fmt_str = "Dataset " + self.__class__.__name__ + "\n"
        fmt_str += "    Number of datapoints: {}\n".format(self.__len__())
        tmp = self.partition
        fmt_str += "    Split: {}\n".format(tmp)
        fmt_str += "    Root Location: {}\n".format(self.root)
        return fmt_str

    def __len__(self):
        return len(self.data)

    def load_data(self):
        return self.data, self.targets

    def get_targets(self, label_file):
        """Get dictionary of targets specified by user."""
        targets = np.load(os.path.join(self.root, label_file))

        tasks = {}
        if self.subsite:
            tasks["subsite"] = targets[:, 0]
        if self.laterality:
            tasks["laterality"] = targets[:, 1]
        if self.behavior:
            tasks["behavior"] = targets[:, 2]
        if self.grade:
            tasks["grade"] = targets[:, 3]

        return tasks

    def __getitem__(self, idx):
        """
        Parameters
        ----------
        index : int
          Index of the data to be loaded.

        Returns
        -------
        (document, target) : tuple
           where target is index of the target class.
        """
        document = self.data[idx]

        if self.transform is not None:
            document = self.transform(document)

        targets = {}
        for key, value in self.targets.items():
            subset = value[idx]

            if self.target_transform is not None:
                subset = self.target_transform(subset)

            targets[key] = subset

        return document, targets


class Vocabulary:
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Tokenizer:
    def __init__(self, train, valid):
        self.vocab = Vocabulary()
        self.train = self.tokenize(train)
        self.valid = self.tokenize(valid)
        self.inverse_tokenize()

    def tokenize(self, data):
        """Tokenize a dataset"""
        # Build the vocabulary
        for doc in tqdm(data):
            for token in doc:
                self.vocab.add_word(token)

        # Tokenize
        idss = []
        for doc in data:
            ids = []
            for token in doc:
                ids.append(self.vocab.word2idx[token])
            idss.append(torch.tensor(ids).type(torch.int64))

        return torch.stack(idss)

    def inverse_tokenize(self):
        self.vocab.inverse = {v: k for k, v in self.vocab.word2idx.items()}


class Egress(Dataset):
    r"""Static split from HJ's data handler

    Targets have six classes, with the following number of classes:

    site: 70,
    subsite: 325,
    laterality: 7,
    histology: 575,
    behaviour: 4,
    grade: 9

    Args:
        root: path to store the data
        split: Split to load. Either 'train' or 'valid'
    """

    store = Path("/gpfs/alpine/proj-shared/med107/NCI_Data/yngtodd/dat.pickle")

    def __init__(self, root, split):
        self._check_split(split)
        self._check_download(root)
        self._load_data(split)
        self._load_vocab()

    def __repr__(self):
        return f"Egress(root={self.root}, split={self.split})"

    def _check_split(self, split):
        assert split in [
            "train",
            "valid",
        ], f"Split must be in {'train', 'valid'}, got {split}"
        self.split = split

    def _check_download(self, root):
        self.root = Path(root)
        if not self.root.exists():
            self._download()

    def _download(self):
        raw = self.root.joinpath("raw")
        raw.mkdir(parents=True)
        raw_data = raw.joinpath("raw.pickle")
        shutil.copy(self.store, raw_data)
        self._preprocess(raw_data)

    def _preprocess(self, raw_data):
        print("Preprocessing data...")
        self._make_processed_dirs()

        with open(raw_data, "rb") as f:
            x_train = np.flip(pickle.load(f), 1)
            y_train = pickle.load(f)
            x_valid = np.flip(pickle.load(f), 1)
            y_valid = pickle.load(f)

        corpus = Tokenizer(x_train, x_valid)
        self.num_vocab = len(corpus.vocab)

        self._save_split("train", corpus.train, y_train)
        self._save_split("valid", corpus.valid, y_valid)
        self._save_vocab(corpus.vocab)
        print("Done!")

    def _save_split(self, split, data, target):
        target = self._create_target(target)
        split_path = self.root.joinpath(f"processed/{split}")
        torch.save(data, split_path.joinpath("data.pt"))
        torch.save(target, split_path.joinpath("target.pt"))

    def _save_vocab(self, vocab):
        torch.save(vocab, self.root.joinpath("vocab.pt"))

    def _make_processed_dirs(self):
        processed = self.root.joinpath("processed")
        processed.joinpath("train").mkdir(parents=True)
        processed.joinpath("valid").mkdir()

    def _create_target(self, arry):
        r"""Convert target dictionary"""
        target = {
            "site": arry[:, 0],
            "subsite": arry[:, 1],
            "laterality": arry[:, 2],
            "histology": arry[:, 3],
            "behaviour": arry[:, 4],
            "grade": arry[:, 5],
        }

        return {
            task: torch.tensor(arry, dtype=torch.long) for task, arry in target.items()
        }

    def _load_data(self, split):
        split_path = self.root.joinpath(f"processed/{split}")
        self.data = torch.load(split_path.joinpath("data.pt"))
        self.target = torch.load(split_path.joinpath("target.pt"))

    def _load_vocab(self):
        self.vocab = torch.load(self.root.joinpath("vocab.pt"))
        self.num_vocab = len(self.vocab)

    def _index_target(self, idx):
        return {task: target[idx] for task, target in self.target.items()}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self._index_target(idx)
