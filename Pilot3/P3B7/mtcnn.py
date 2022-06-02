import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict
from dataclasses import dataclass


@dataclass
class Hparams:
    kernel1: int = 3
    kernel2: int = 4
    kernel3: int = 5
    embed_dim: int = 300
    n_filters: int = 300
    sent_len: int = 512
    vocab_size: int = 10_000


class Conv1dPool(nn.Module):
    """ Conv1d => AdaptiveMaxPool1d => Relu """

    def __init__(self, embedding_dim: int, n_filters: int, kernel_size: int):
        super(Conv1dPool, self).__init__()
        self.conv = nn.Conv1d(embedding_dim, n_filters, kernel_size)
        self._weight_init()

    def _weight_init(self):
        """ Initialize the convolution weights """
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.conv.weight, gain)

    def forward(self, x):
        x = self.conv(x)
        x = F.adaptive_max_pool1d(x, output_size=1)
        x = F.relu(x)
        return x


class MultitaskClassifier(nn.Module):
    """ Multi-task Classifier
    Args:
        input_dim: input dimension for each of the linear layers
        tasks: dictionary of tasks and their respective number of classes
    """

    def __init__(self, input_dim: int, tasks: Dict[str, int]):
        super(MultitaskClassifier, self).__init__()
        self.tasks = tasks

        for task, num_classes in tasks.items():
            self.add_module(
                task,
                nn.Linear(input_dim, num_classes)
            )

    def num_classes(self, task):
        """ Get number of classes for a task. """
        return self.tasks[task]

    def forward(self, x):
        logits = {}
        for task, _ in self.tasks.items():
            logits[task] = self._modules[task](x)

        return logits


class MTCNN(nn.Module):
    """ Multi-task CNN a la Yoon Kim
    Args:
        tasks: dictionary of tasks and their respective number of classes.
               This is used by the MultitaskClassifier.
        hparams: dataclass of the model hyperparameters
    """

    def __init__(self, tasks: Dict[str, int], hparams: Hparams):
        super(MTCNN, self).__init__()
        self.hparams = hparams
        self.embed = nn.Embedding(hparams.vocab_size, hparams.embed_dim)
        self.conv1 = Conv1dPool(hparams.embed_dim, hparams.n_filters, hparams.kernel1)
        self.conv2 = Conv1dPool(hparams.embed_dim, hparams.n_filters, hparams.kernel2)
        self.conv3 = Conv1dPool(hparams.embed_dim, hparams.n_filters, hparams.kernel3)
        self.classifier = MultitaskClassifier(self._filter_sum(), tasks)
        self._weight_init()

    def _weight_init(self):
        """ Initialize the network weights """
        self._embed_init()

    def _embed_init(self, initrange=0.05):
        """ Initialize the embedding weights """
        nn.init.uniform_(self.embed.weight, -initrange, initrange)

    def _filter_sum(self):
        return self.hparams.n_filters * 3

    def loss_value(self, y_pred, y_true, reduce="sum"):
        """ Compute the cross entropy loss """
        losses = {}

        for key, value in y_true.items():
            losses[key] = F.cross_entropy(y_pred[key], y_true[key])

        if reduce:
            total = 0
            for _, loss in losses.items():
                total += loss

            if reduce == "mean":
                losses = total / len(losses)
            elif reduce == "sum":
                losses = total

        return losses

    def forward(self, x):
        x = self.embed(x)
        # Make sure embed_dim is the channel dim for convolution
        x = x.transpose(1, 2)

        conv_results = []
        conv_results.append(self.conv1(x).view(-1, self.hparams.n_filters))
        conv_results.append(self.conv2(x).view(-1, self.hparams.n_filters))
        conv_results.append(self.conv3(x).view(-1, self.hparams.n_filters))
        x = torch.cat(conv_results, 1)

        logits = self.classifier(x)
        return logits
