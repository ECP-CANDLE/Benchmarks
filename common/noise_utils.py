import random
from keras.utils import np_utils


def label_flip(y_data, y_noise_level):
    flip_count = 0
    y_data_categorical = np_utils.to_categorical(y_data)
    for i in range(0, y_data_categorical.shape[0]):
        if random.random() < y_noise_level:
            flip_count += 1
            for j in range(y_data_categorical.shape[1]):
                y_data_categorical[i][j] = int(not y_data_categorical[i][j])

    y_noise_generated = float(flip_count) / float(y_data.shape[0])
    print("Uncorrelated label noise generation:\n")
    print("Labels flipped on {} samples out of {}: {:06.4f} ({:06.4f} requested)\n".format(
          flip_count, y_data.shape[0], y_noise_generated, y_noise_level))

    return y_data_categorical, y_noise_generated

def label_flip_correlated(y_data, y_noise_level, x_data, feature_column, threshold):
    flip_count = 0
    y_data_categorical = np_utils.to_categorical(y_data)
    for i in range(0, y_data_categorical.shape[0]):
        if x_data[i][feature_column] > threshold:
            if random.random() < y_noise_level:
                flip_count += 1
                for j in range(y_data_categorical.shape[1]):
                    y_data_categorical[i][j] = int(not y_data_categorical[i][j])

    y_noise_generated = float(flip_count) / float(y_data.shape[0])
    print("Correlated label noise generation for feature {:d}:\n".format(feature_column))
    print("Labels flipped on {} samples out of {}: {:06.4f} ({:06.4f} requested)\n".format(
          flip_count, y_data.shape[0], y_noise_generated, y_noise_level))

    return y_data_categorical, y_noise_generated
