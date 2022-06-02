import numpy as np
from typing import Dict
from torch.utils.data import Dataset


class RandomData(Dataset):
    """ Random dataset - Useful for quick iterating """

    def __init__(self, x_dim: int, num_samples: int, tasks: Dict[str, int], seed: int = 13):
        np.random.seed(seed)
        self.data = self.create_data(x_dim, num_samples)
        self.labels = self.create_labels(tasks, num_samples)

    def create_data(self, x_dim, num_samples):
        data = [np.random.randn(x_dim).astype('f') for _ in range(num_samples)]
        return np.stack(data)

    def create_labels(self, tasks, num_samples):
        labels = {}
        for task, num_classes in tasks.items():
            labels[task] = np.random.randint(num_classes, size=num_samples)

        return labels

    def index_labels(self, idx):
        """ Index into the labels """
        return {key: value[idx] for key, value in self.labels.items()}

    def load_data(self):
        return self.data, self.labels

    def __repr__(self):
        return f'Random supervised dataset'

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.index_labels(idx)
