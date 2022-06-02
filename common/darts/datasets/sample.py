from sklearn.utils import resample
from darts.api.dataset import Subset


def dummy_indices(dataset):
    """ Get indexes for the dataset """
    return [x for x in range(len(dataset))]


def sample(dataset, num_samples, replace=True):
    """ Sample the dataset """
    data_idx = dummy_indices(dataset)
    sample_idx = resample(data_idx, n_samples=num_samples, replace=replace)
    return Subset(dataset, sample_idx)
