import random
import numpy as np


def add_noise_new(data, labels, params):
    # new refactoring of the noise injection methods
    # noise_mode sets the pattern of the noise injection
    #   cluster: apply to the samples and features defined by noise_samples and noise_features
    #   conditional : apply to features conditional on a threshold
    #
    # noise_type sets the form of the noise
    #   gaussian: Gaussian feature noise with noise_scale as std_dev
    #   uniform: Uniformly distributed noise on the interval [-noise_scale, noise_scale]
    #   label: Flip labels
    if params['noise_injection']:
        if params['label_noise']:
            # check if we want noise correlated with a feature
            if params['noise_correlated']:
                labels, y_noise_gen = label_flip_correlated(labels,
                                                            params['label_noise'], data,
                                                            params['feature_col'],
                                                            params['feature_threshold'])
            # else add uncorrelated noise
            else:
                labels, y_noise_gen = label_flip(labels, params['label_noise'])
        # check if noise is on for RNA-seq data
        elif params['noise_gaussian']:
            data = add_gaussian_noise(data, 0, params['std_dev'])
        elif params['noise_cluster']:
            data = add_cluster_noise(data, loc=0., scale=params['std_dev'],
                                     col_ids=params['feature_col'],
                                     noise_type=params['noise_type'],
                                     row_ids=params['sample_ids'],
                                     y_noise_level=params['label_noise'])
        elif params['noise_column']:
            data = add_column_noise(data, 0, params['std_dev'],
                                    col_ids=params['feature_col'],
                                    noise_type=params['noise_type'])

    return data, labels


def add_noise(data, labels, params):
    # put all the logic under the add_noise switch
    if params['add_noise']:
        if params['label_noise']:
            # check if we want noise correlated with a feature
            if params['noise_correlated']:
                labels, y_noise_gen = label_flip_correlated(labels,
                                                            params['label_noise'], data,
                                                            params['feature_col'],
                                                            params['feature_threshold'])
            # else add uncorrelated noise
            else:
                labels, y_noise_gen = label_flip(labels, params['label_noise'])
        # check if noise is on for RNA-seq data
        elif params['noise_gaussian']:
            data = add_gaussian_noise(data, 0, params['std_dev'])
        elif params['noise_cluster']:
            data = add_cluster_noise(data, loc=0., scale=params['std_dev'],
                                     col_ids=params['feature_col'],
                                     noise_type=params['noise_type'],
                                     row_ids=params['sample_ids'],
                                     y_noise_level=params['label_noise'])
        elif params['noise_column']:
            data = add_column_noise(data, 0, params['std_dev'],
                                    col_ids=params['feature_col'],
                                    noise_type=params['noise_type'])

    return data, labels


def label_flip(y_data_categorical, y_noise_level):
    flip_count = 0
    for i in range(0, y_data_categorical.shape[0]):
        if random.random() < y_noise_level:
            flip_count += 1
            for j in range(y_data_categorical.shape[1]):
                y_data_categorical[i][j] = int(not y_data_categorical[i][j])

    y_noise_generated = float(flip_count) / float(y_data_categorical.shape[0])
    print("Uncorrelated label noise generation:\n")
    print("Labels flipped on {} samples out of {}: {:06.4f} ({:06.4f} requested)\n".format(
          flip_count, y_data_categorical.shape[0], y_noise_generated, y_noise_level))

    return y_data_categorical, y_noise_generated


def label_flip_correlated(y_data_categorical, y_noise_level, x_data, col_ids, threshold):
    for col_id in col_ids:
        flip_count = 0
        for i in range(0, y_data_categorical.shape[0]):
            if x_data[i][col_id] > threshold:
                if random.random() < y_noise_level:
                    print(i, y_data_categorical[i][:])
                    flip_count += 1
                    for j in range(y_data_categorical.shape[1]):
                        y_data_categorical[i][j] = int(not y_data_categorical[i][j])
                    print(i, y_data_categorical[i][:])

        y_noise_generated = float(flip_count) / float(y_data_categorical.shape[0])
        print("Correlated label noise generation for feature {:d}:\n".format(col_id))
        print("Labels flipped on {} samples out of {}: {:06.4f} ({:06.4f} requested)\n".format(
            flip_count, y_data_categorical.shape[0], y_noise_generated, y_noise_level))

    return y_data_categorical, y_noise_generated


# Add simple Gaussian noise to RNA seq values, assume normalized x data
def add_gaussian_noise(x_data, loc=0., scale=0.5):
    print("added gaussian noise")
    train_noise = np.random.normal(loc, scale, size=x_data.shape)
    x_data = x_data + train_noise
    return x_data


# Add simple Gaussian noise to a list of RNA seq values, assume normalized x data
def add_column_noise(x_data, loc=0., scale=0.5, col_ids=[0], noise_type='gaussian'):
    for col_id in col_ids:
        print("added", noise_type, "noise to column ", col_id)
        print(x_data[:, col_id].T)
        if noise_type == 'gaussian':
            train_noise = np.random.normal(loc, scale, size=x_data.shape[0])
        elif noise_type == 'uniform':
            train_noise = np.random.uniform(-1.0 * scale, scale, size=x_data.shape[0])
        print(train_noise)
        x_data[:, col_id] = 1.0 * x_data[:, col_id] + 1.0 * train_noise.T
        print(x_data[:, col_id].T)
    return x_data


# Add noise to a list of RNA seq values, for a fraction of samples assume normalized x data
def add_cluster_noise(x_data, loc=0., scale=0.5, col_ids=[0], noise_type='gaussian', row_ids=[0], y_noise_level=0.0):
    # loop over all samples
    num_samples = len(row_ids)
    flip_count = 0
    for row_id in row_ids:
        # only perturb a fraction of the samples
        if random.random() < y_noise_level:
            flip_count += 1
            for col_id in col_ids:
                print("added", noise_type, "noise to row, column ", row_id, col_id)
                print(x_data[row_id, col_id])
                if noise_type == 'gaussian':
                    train_noise = np.random.normal(loc, scale)
                elif noise_type == 'uniform':
                    train_noise = np.random.uniform(-1.0 * scale, scale)
                print(train_noise)
                x_data[row_id, col_id] = 1.0 * x_data[row_id, col_id] + 1.0 * train_noise
                print(x_data[row_id, col_id])

    y_noise_generated = float(flip_count) / float(num_samples)
    print("Noise added to {} samples out of {}: {:06.4f} ({:06.4f} requested)\n".format(
        flip_count, num_samples, y_noise_generated, y_noise_level))
    return x_data
