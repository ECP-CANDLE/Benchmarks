from __future__ import print_function

import itertools

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


np.set_printoptions(precision=2)


def plot_ROC(fpr_keras, tpr_keras, auc_keras, fname, xlabel_add='', ylabel_add='', zoom=False):

    plt.figure()
    if zoom:
        plt.xlim(0, 0.2)
        plt.ylim(0.8, 1)

    plt.plot([0, 1], [0, 1], 'k--', label='No Skill')
    plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
    plt.xlabel('False positive rate' + xlabel_add)
    plt.ylabel('True positive rate' + ylabel_add)
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.savefig(fname, bbox_inches='tight')
    plt.close()


def plot_RF(recall_keras, precision_keras, pr_keras, no_skill, fname, xlabel_add='', ylabel_add=''):

    plt.figure()
    plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
    plt.plot(recall_keras, precision_keras, label='PR Keras (area = {:.3f})'.format(pr_keras))
    plt.xlabel('Recall' + xlabel_add)
    plt.ylabel('Precision' + ylabel_add)
    plt.title('PR curve')
    plt.legend(loc='best')
    plt.savefig(fname, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(cm, fname, classes, normalize=False, title='Confusion matrix'):
    """
    This function prints and plots the confusion matrix. Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print('Normalized confusion matrix')
    else:
        print('Confusion matrix, without normalization')
    print(cm)

    cmap = plt.cm.Blues
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(fname, bbox_inches='tight')
    plt.close()
