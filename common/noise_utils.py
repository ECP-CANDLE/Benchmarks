import random
import numpy as np
from keras.utils import np_utils


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

def label_flip_correlated(y_data_categorical, y_noise_level, x_data, feature_column, threshold):
    flip_count = 0
    for i in range(0, y_data_categorical.shape[0]):
        if x_data[i][feature_column] > threshold:
            if random.random() < y_noise_level:
                flip_count += 1
                for j in range(y_data_categorical.shape[1]):
                    y_data_categorical[i][j] = int(not y_data_categorical[i][j])

    y_noise_generated = float(flip_count) / float(y_data_categorical.shape[0])
    print("Correlated label noise generation for feature {:d}:\n".format(feature_column))
    print("Labels flipped on {} samples out of {}: {:06.4f} ({:06.4f} requested)\n".format(
          flip_count, y_data_categorical.shape[0], y_noise_generated, y_noise_level))

    return y_data_categorical, y_noise_generated

# Add simple Gaussian noise to RNA seq values, assume normalized x data
def add_gaussian_noise(x_data, loc=0., scale=0.5):
    print("added gaussian noise")
    train_noise = np.random.normal(loc, scale, size=x_data.shape)
    x_data = x_data + train_noise 
    return x_data
