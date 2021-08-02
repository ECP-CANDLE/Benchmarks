from abc import abstractmethod
import pandas as pd


class Dataset:
    """ Abstract dataset - Used for both Keras and Pytorch"""

    @abstractmethod
    def __getitem__(self, idx):
        """Gets batch at position `index`.
        Parameters
        ----------
        idx: index position of the batch in the data.
        Returns
        -------
        A batch
        """
        raise NotImplementedError

    @abstractmethod
    def __len__(self):
        """Length of the dataset.
        Returns
        -------
        The number of samples in the data.
        """
        raise NotImplementedError

    def on_epoch_end(self):
        """ Keras method called at the end of every epoch. """
        pass

    def __iter__(self):
        """Create a generator that iterates over the data."""
        for item in (self[i] for i in range(len(self))):
            yield item


class InMemoryDataset(Dataset):
    """ Abstract class for in memory data """

    def load_data(self):
        """ Load data and labels """
        raise NotImplementedError

    def dataframe(self):
        """ Load the data as a pd.DataFrame """
        data, labels = self.load_data()

        if isinstance(labels, dict):
            # We are in the multitask case
            data_dict = {'data': data}
            for key, value in labels.items():
                data_dict[key] = value
        else:
            data_dict = {'data': data, 'labels': labels}

        return pd.DataFrame(data_dict)

    def to_csv(self, path):
        """ Save the data to disk """
        self.dataframe().to_csv(path, index=False)

    def load_cached(self, path):
        """ Load the data from disk """
        frame = pd.read_csv(path)

        self.data = frame.pop('data')

        if len(frame.columns) > 1:
            self.labels = frame.to_dict()
        else:
            self.labels = frame['labels']


class Subset(InMemoryDataset):
    """Subset of a dataset at specified indices.

    Args:
        dataset (Dataset): The dataset to be subsetted
        indices (sequence): Indices in the whole set selected for subset
    """
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)

    def load_data(self):
        return self.dataset[self.indices]
