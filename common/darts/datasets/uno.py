import os
import torch

import numpy as np
import pandas as pd

from darts.api import InMemoryDataset
from darts.datasets.utils import (
    download_url, makedir_exist_ok
)


class Uno(InMemoryDataset):
    """Uno Dataset

    Parameters
    ----------
    root str :
        Root directory of dataset where ``processed/training.npy``
        ``processed/validation.npy and ``processed/test.npy`` exist.

    partition : str
        dataset partition to be loaded.
        Either 'train', 'validation', or 'test'.

    download : bool, optional
        If true, downloads the dataset from the internet and
        puts it in root directory. If dataset is already downloaded, it is not
        downloaded again.
    """
    urls = [
        'http://ftp.mcs.anl.gov/pub/candle/public/benchmarks/Pilot1/uno/top_21_auc_1fold.uno.h5',
    ]

    training_data_file = 'train_data.pt'
    training_label_file = 'train_labels.pt'
    test_data_file = 'test_data.pt'
    test_label_file = 'test_labels.pt'

    def __init__(self, root, partition, transform=None,
                 target_transform=None, download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.'
                               + ' You can use download=True to download it')

        self.partition = partition
        if self.partition == 'train':
            data_file = self.training_data_file
            label_file = self.training_label_file
        elif self.partition == 'test':
            data_file = self.test_data_file
            label_file = self.test_label_file
        else:
            raise ValueError("Partition must either be 'train' or 'test'.")

        self.data = torch.load(os.path.join(self.processed_folder, data_file))
        self.targets = torch.load(os.path.join(self.processed_folder, label_file))

    def __len__(self):
        return len(self.data['gene_data'])

    def load_data(self):
        return self.data, self.targets

    def read_data(self, data_file, partition):
        """ Read in the H5 data """
        if partition == 'train':
            gene_data = 'x_train_0'
            drug_data = 'x_train_1'
        else:
            gene_data = 'x_val_0'
            drug_data = 'x_val_1'

        gene_data = torch.tensor(pd.read_hdf(data_file, gene_data).values)
        drug_data = torch.tensor(pd.read_hdf(data_file, drug_data).values)
        data = {'gene_data': gene_data, 'drug_data': drug_data}

        return data

    def read_targets(self, data_file, partition):
        """Get dictionary of targets specified by user."""
        if partition == 'train':
            label = 'y_train'
        else:
            label = 'y_val'

        tasks = {
            'response': torch.tensor(
                pd.read_hdf(data_file, label)['AUC'].apply(lambda x: 1 if x < 0.5 else 0)
            )
        }

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
        data = self.data['gene_data'][idx]

        if self.transform is not None:
            data = self.transform(data)

        targets = {}
        for key, value in self.targets.items():
            subset = value[idx]

            if self.target_transform is not None:
                subset = self.target_transform(subset)

            targets[key] = subset

        return data, targets

    @property
    def raw_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'raw')

    @property
    def processed_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'processed')

    def _check_exists(self):
        return os.path.exists(os.path.join(self.processed_folder, self.training_data_file)) and \
            os.path.exists(os.path.join(self.processed_folder, self.training_label_file)) and \
            os.path.exists(os.path.join(self.processed_folder, self.test_data_file)) and \
            os.path.exists(os.path.join(self.processed_folder, self.test_label_file))

    @staticmethod
    def extract_array(path, remove_finished=False):
        print('Extracting {}'.format(path))
        arry = np.load(path)
        if remove_finished:
            os.unlink(path)

    def download(self):
        """Download the Synthetic data if it doesn't exist in processed_folder already."""

        if self._check_exists():
            return

        makedir_exist_ok(self.raw_folder)
        makedir_exist_ok(self.processed_folder)

        # download files
        for url in self.urls:
            filename = url.rpartition('/')[2]
            file_path = os.path.join(self.raw_folder, filename)
            download_url(url, root=self.raw_folder, filename=filename, md5=None)
            # self.extract_array(path=file_path, remove_finished=False)

        # process and save as numpy files
        print('Processing...')

        training_set = (
            self.read_data(os.path.join(self.raw_folder, 'top_21_auc_1fold.uno.h5'), 'train'),
            self.read_targets(os.path.join(self.raw_folder, 'top_21_auc_1fold.uno.h5'), 'train')
        )
        test_set = (
            self.read_data(os.path.join(self.raw_folder, 'top_21_auc_1fold.uno.h5'), 'test'),
            self.read_targets(os.path.join(self.raw_folder, 'top_21_auc_1fold.uno.h5'), 'test')
        )

        # Save processed training data
        train_data_path = os.path.join(self.processed_folder, self.training_data_file)
        torch.save(training_set[0], train_data_path)
        train_label_path = os.path.join(self.processed_folder, self.training_label_file)
        torch.save(training_set[1], train_label_path)

        # Save processed test data
        test_data_path = os.path.join(self.processed_folder, self.test_data_file)
        torch.save(test_set[0], test_data_path)
        test_label_path = os.path.join(self.processed_folder, self.test_label_file)
        torch.save(test_set[1], test_label_path)

        print('Done!')

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = self.partition
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        return fmt_str
