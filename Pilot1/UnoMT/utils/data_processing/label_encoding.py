"""
    File Name:          UnoPytorch/label_encoding.py
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               8/13/18
    Python Version:     3.6.6
    File Description:
        This file includes helper functions that are re
"""

import json
import logging
import os
import numpy as np


logger = logging.getLogger(__name__)

# Folders for raw/processed data
RAW_FOLDER = './raw/'
PROC_FOLDER = './processed/'


def get_label_dict(data_root: str, dict_name: str):
    """label_dict = get_label_list('./data/', 'label_dict.txt')

    Get the encoding dictionary from the given data path.

    Args:
        data_root (str): path to data root folder.
        dict_name (str): label encoding dictionary name.

    Returns:
        dict: encoding dictionary. {} if the dictionary does not exist.
    """
    dict_path = os.path.join(data_root, PROC_FOLDER, dict_name)

    if os.path.exists(dict_path):
        with open(dict_path, 'r') as f:
            label_encoding_dict = json.load(f)
        return label_encoding_dict
    else:
        return {}


def update_label_dict(data_root: str, dict_name: str, new_labels: iter):
    """label_dict = update_label_dict('./data/',
                                      'label_dict.txt',
                                      ['some', 'labels'])

    This function will check if there exists dictionary for label encoding.
        * if not, it construct a new encoding dictionary;
        * otherwise, it will load the existing dictionary and update if not
            all he labels to be encoded are in the dictionary;
    Lastly, it returns the updated encoded dictionary.

    For example, the label encoding dictionary for labels ['A', 'B', 'C'] is
    {'A': 0, 'B': 1, 'C': 2}.

    Note that all the labels should be strings.

    Args:
        data_root (str): path to data root folder.
        dict_name (str): label encoding dictionary name.
        new_labels (iter): iterable structure of labels to be encoded.

    Returns:
        dict: update encoding dictionary for labels.
    """

    label_encoding_dict = get_label_dict(data_root=data_root,
                                         dict_name=dict_name)

    # Get all the old labels and check if we have new ones
    old_labels = [str(line) for line in label_encoding_dict.keys()]

    if len(set(new_labels) - set(old_labels)) != 0:

        logger.debug('Updating encoding dict %s' % dict_name)

        # If not, extend the label encoding dict
        old_idx = len(old_labels)
        for idx, l in enumerate(set(new_labels) - set(old_labels)):
            label_encoding_dict[str(l)] = idx + old_idx

        # Save the dict to corresponding path
        try:
            os.makedirs(os.path.join(data_root, PROC_FOLDER))
        except FileExistsError:
            pass

        dict_path = os.path.join(data_root, PROC_FOLDER, dict_name)
        with open(dict_path, 'w') as f:
            json.dump(label_encoding_dict, f, indent=4)

    return label_encoding_dict


def encode_label_to_int(data_root: str, dict_name: str, labels: iter):
    """encoded_labels = label_encoding('./data/',
                                       'label_dict.txt',
                                       dataframe['column'])

    This function encodes a iterable structure of labels into list of integer.

    Args:
        data_root (str): path to data root folder.
        dict_name (str): label encoding dictionary name.
        labels (iter): an iterable structure of labels to be encoded.

    Returns:
        list: list of integer encoded labels.
    """

    label_encoding_dict = update_label_dict(data_root=data_root,
                                            dict_name=dict_name,
                                            new_labels=labels)
    return [label_encoding_dict[str(s)] for s in labels]


def encode_int_to_onehot(labels: iter, num_classes: int = None):
    """one_hot_labels = encode_int_to_onehot(int_labels, num_classes=10)

    This function converts an iterable structure of integer labels into
    one-hot encoding.

    Args:
        labels (iter): an iterable structure of int labels to be encoded.
        num_classes (int): number of classes for labels. When set to None,
            the function will infer from given labels.

    Returns:
        list: list of one-hot-encoded labels.
    """

    # Infer the number of classes and sanity check
    if num_classes is None:
        if len(set(labels)) != (np.amax(labels) + 1):
            logger.warning('Possible incomplete labels.'
                           'Set the num_classes to ensure the correctness.')

        num_classes = len(set(labels))
    else:
        assert num_classes >= len(set(labels))

    # Convert the labels into one-hot-encoded ones
    encoded_labels = []
    for label in labels:
        encoded = [0] * num_classes
        encoded[label] = 1
        encoded_labels.append(encoded)

    return encoded_labels
