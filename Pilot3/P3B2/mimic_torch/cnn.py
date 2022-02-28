import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import numpy as np

###############################################################################
# Helper functions
###############################################################################


def tokenize_data(input_data, vocab):
    tokenized_data = []

    for row in input_data:

        current = []

        rowSplit = row.split()
        for s in rowSplit:
            if s in vocab:
                current.append(vocab[s])

        tokenized_data.append(np.array(current, dtype=np.int32))

    return tokenized_data


def generate_torch_data(input_data, selected_indices):

    if len(input_data) < 1:
        return None

    maxLength = 5
    for i in selected_indices:
        g = input_data[i]
        if len(g) > maxLength:
            maxLength = len(g)

    tokens = np.zeros((len(selected_indices), maxLength), dtype=np.long)

    for i in range(len(selected_indices)):
        ind = selected_indices[i]
        g = input_data[ind]
        if len(g) > 0:
            tokens[i, :len(g)] = g[:]

    return torch.LongTensor(tokens)

###############################################################################
# MTCNN model
###############################################################################


@dataclass
class Hparams:
    kernel1: int = 3
    kernel2: int = 4
    kernel3: int = 5
    embed_dim: int = 300
    n_filters: int = 300
    vocab_size: int = 22275
    fixed_embeddings: bool = False


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


class Model(nn.Module):

    def __init__(self, numberOfClasses, hparams: Hparams, embeddings=None):
        super(Model, self).__init__()
        self.hparams = hparams

        if (hparams.fixed_embeddings) and (embeddings is not None):
            self.embed = nn.Embedding.from_pretrained(embeddings, freeze=True)
        else:
            self.embed = nn.Embedding(hparams.vocab_size, hparams.embed_dim, padding_idx=0)
            self._embed_init()

        self.conv1 = Conv1dPool(hparams.embed_dim, hparams.n_filters, hparams.kernel1)
        self.conv2 = Conv1dPool(hparams.embed_dim, hparams.n_filters, hparams.kernel2)
        self.conv3 = Conv1dPool(hparams.embed_dim, hparams.n_filters, hparams.kernel3)
        self.classifier = nn.Linear(self._filter_sum(), numberOfClasses)

    def _embed_init(self, initrange=0.05):
        """ Initialize the embedding weights """
        nn.init.uniform_(self.embed.weight, -initrange, initrange)

    def _filter_sum(self):
        return self.hparams.n_filters * 3

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
