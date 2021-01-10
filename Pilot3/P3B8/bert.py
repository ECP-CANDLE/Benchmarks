import torch
import torch.nn as nn

from pytorch_pretrained_bert import (
    BertForSequenceClassification
)


class HiBERT(nn.Module):

    def __init__(self, bert_load_path, num_classes):
        super(HiBERT, self).__init__()

        self.bert = BertForSequenceClassification.from_pretrained(
            bert_load_path, num_labels=num_classes
        )

    def forward(self, input_ids, input_mask, segment_ids):
        return self.bert(input_ids, input_mask, segment_ids, labels=None)
