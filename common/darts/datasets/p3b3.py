import os
import numpy as np
from torch.utils.data import Dataset


class P3B3(Dataset):
    """P3B3 Synthetic Dataset.

    Args:
        root: str
            Root directory of dataset where CANDLE loads P3B3 data.

        partition: str
            dataset partition to be loaded.
            Must be either 'train' or 'test'.
    """
    training_data_file = 'train_X.npy'
    training_label_file = 'train_Y.npy'
    test_data_file = 'test_X.npy'
    test_label_file = 'test_Y.npy'

    def __init__(self, root, partition, subsite=True,
                 laterality=True, behavior=True, grade=True,
                 transform=None, target_transform=None):
        self.root = root
        self.partition = partition
        self.transform = transform
        self.target_transform = target_transform
        self.subsite = subsite
        self.laterality = laterality
        self.behavior = behavior
        self.grade = grade

        if self.partition == 'train':
            data_file = self.training_data_file
            label_file = self.training_label_file
        elif self.partition == 'test':
            data_file = self.test_data_file
            label_file = self.test_label_file
        else:
            raise ValueError("Partition must either be 'train' or 'test'.")

        self.data = np.load(os.path.join(self.root, data_file))
        self.targets = self.get_targets(label_file)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = self.partition
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
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
            tasks['subsite'] = targets[:, 0]
        if self.laterality:
            tasks['laterality'] = targets[:, 1]
        if self.behavior:
            tasks['behavior'] = targets[:, 2]
        if self.grade:
            tasks['grade'] = targets[:, 3]

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
