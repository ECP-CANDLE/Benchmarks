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

    def forward(self, input_ids, input_mask, segment_ids, n_segs):

        n_segs = n_segs.view(-1)
        input_ids_ = input_ids.view(-1, 512)[:n_segs]
        input_mask_ = input_mask.view(-1, 512)[:n_segs]
        segment_ids_ = segment_ids.view(-1, 512)[:n_segs]

        logits = self.bert(input_ids_, input_mask_, segment_ids_, labels=None)
        logits = torch.max(logits, 0)[0]

        return logits

