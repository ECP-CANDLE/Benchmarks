import torch
from torch.utils.data import Dataset


class MimicDatasetSynthetic(Dataset):

    def __init__(self,
            doc_length=512,
            num_vocab=10_000,
            num_docs=100,
            num_classes=10
        ):

        self.doc_length = doc_length
        self.num_vocab = num_vocab
        self.num_docs = num_docs
        self.num_classes = num_classes

        self.docs = self.create_docs(doc_length, num_vocab, num_docs)
        self.masks = self.create_masks(doc_length, num_docs)
        self.segment_ids = self.create_segment_ids(doc_length, num_docs)
        self.labels = self.create_labels(num_classes, num_docs)

    def __repr__(self):
        return (
            f'MimicRandom(doc_length={self.doc_length}, '
            f'num_vocab={self.num_vocab}, '
            f'num_docs={self.num_docs}, '
            f'num_classes={self.num_classes}) '
        )

    def __len__(self):
        return self.num_docs

    def __getitem__(self, idx):
        return {
            "tokens": self.docs[idx],
            "masks": self.masks[idx],
            "seg_ids": self.segment_ids[idx],
            "label": self.labels[idx],
        }

    def random_doc(self, length, num_vocab):
        return torch.LongTensor(length).random_(0, num_vocab+1)

    def create_docs(self, length, num_vocab, num_docs):
        docs = [self.random_doc(length, num_vocab) for _ in range(num_docs)]
        return torch.stack(docs)

    def random_mask(self, length):
        return torch.LongTensor(length).random_(0, 2)

    def create_masks(self, length, num_docs):
        masks = [self.random_mask(length) for _ in range(num_docs)]
        return torch.stack(masks)

    def empty_segment_id(self, length):
        return torch.zeros(length, dtype=torch.long)

    def create_segment_ids(self, length, num_docs):
        segment_ids = [self.empty_segment_id(length) for _ in range(num_docs)]
        return torch.stack(segment_ids)

    def random_multitask_label(self, num_classes):
        return torch.FloatTensor(num_classes).random_(0, 2)

    def create_labels(self, num_classes, num_docs):
        labels = [self.random_multitask_label(num_classes) for _ in range(num_docs)]
        return torch.stack(labels)

