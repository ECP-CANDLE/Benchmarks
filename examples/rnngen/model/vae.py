import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CharRNN(nn.Module):
    def __init__(self, vocab_size, emb_size, max_len=150):
        super(CharRNN, self).__init__()
        self.max_len = max_len
        self.emb = nn.Embedding(vocab_size, emb_size)
        self.lstm = nn.LSTM(emb_size, 256, dropout=0.3, num_layers=2)
        self.linear = nn.Linear(256, vocab_size)

    # pass x as a pack padded sequence please.
    def forward(self, x, with_softmax=False):
        # do stuff to train
        dv = x[0].device
        x = [self.emb(torch.from_numpy(np.flip(x_.cpu().numpy())).to(dv)) for x_ in x]

        x = nn.utils.rnn.pack_sequence(x, enforce_sorted=False)

        x, _ = self.lstm(x)

        x, lens = nn.utils.rnn.pad_packed_sequence(x, padding_value=0, total_length=self.max_len)
        x = self.linear(x)

        if with_softmax:
            return F.softmax(x, dim=-1)
        else:
            return x

    def sample(self):
        return None
