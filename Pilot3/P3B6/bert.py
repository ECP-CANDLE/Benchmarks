import torch
import torch.nn as nn

from transformers import (
    BertForSequenceClassification, BertConfig
)


class HiBERT(nn.Module):

    def __init__(self, num_labels):
        super(HiBERT, self).__init__()

        self.bert = BertForSequenceClassification(
            BertConfig(
                num_labels,
                hidden_size=128,
                num_attention_heads=2,
                num_hidden_layers=2
            )
        )

    def forward(self, input_ids, input_mask=None, segment_ids=None, labels=None):
        return self.bert(input_ids, input_mask, segment_ids, labels=labels)
