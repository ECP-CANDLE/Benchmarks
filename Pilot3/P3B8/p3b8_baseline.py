import os
import time
import torch

import p3b8 as bmk
import candle

import numpy as np

import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from sklearn.metrics import f1_score

from transformers import (
    BertForSequenceClassification, BertConfig
)

from random_data import MimicDatasetSynthetic


def initialize_parameters():
    """ Initialize the parameters for the P3B5 benchmark """

    p3b8_bench = bmk.BenchmarkP3B8(
        bmk.file_path,
        "default_model.txt",
        "pytorch",
        prog="p3b8",
        desc="BERT Quantized",
    )

    gParameters = candle.finalize_parameters(p3b8_bench)
    return gParameters


def load_data(args):
    """ Initialize random data

    Args:
        gParameters: parameters from candle

    Returns:
        train, valid, test sets
    """
    num_classes = args.num_classes
    num_train_samples = args.num_train_samples
    num_valid_samples = args.num_valid_samples
    num_test_samples = args.num_test_samples

    train = MimicDatasetSynthetic(num_docs=num_train_samples, num_classes=num_classes)
    valid = MimicDatasetSynthetic(num_docs=num_valid_samples, num_classes=num_classes)
    test = MimicDatasetSynthetic(num_docs=num_test_samples, num_classes=num_classes)

    return train, valid, test


def create_data_loaders(args):
    """ Initialize data loaders

    Args:
        gParameters: parameters from candle

    Returns:
        train, valid, test data loaders
    """
    train, valid, test = load_data(args)
    train_loader = DataLoader(train, batch_size=args.batch_size)
    valid_loader = DataLoader(valid, batch_size=args.batch_size)
    test_loader = DataLoader(test, batch_size=args.batch_size)
    return train_loader, valid_loader, test_loader


def train(dataloader, model, optimizer, criterion, args, epoch):
    model.train()

    for idx, batch in enumerate(dataloader):
        train_loss = 0.0
        optimizer.zero_grad()

        input_ids = batch["tokens"].to(args.device)
        labels = batch["label"].to(args.device)

        output = model(
            input_ids,
            labels=labels
        )

        output.loss.backward()
        optimizer.step()

        print(f"epoch: {epoch}, batch: {idx}, train loss: {output.loss}")


def validate(dataloader, model, args, device, epoch):
    model.eval()

    with torch.no_grad():
        for idx, batch in enumerate(dataloader):

            input_ids = batch["tokens"].to(device)
            labels = batch["label"].to(args.device)

            output = model(
                input_ids,
                labels=labels
            )

            print(f"epoch: {epoch}, batch: {idx}, valid loss: {output.loss}")


def time_evaluation(dataloader, model, args, device):
    s = time.time()
    loss = validate(dataloader, model, args, device, epoch=0)
    elapsed = time.time() - s
    print(f"\telapsed time (seconds): {elapsed:.1f}")


def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p") / 1e6)
    os.remove('temp.p')


def run(args):
    args = candle.ArgumentStruct(**args)
    args.cuda = torch.cuda.is_available()
    args.device = torch.device(f"cuda" if args.cuda else "cpu")

    train_loader, valid_loader, test_loader = create_data_loaders(args)

    config = BertConfig(
        num_attention_heads=2,
        hidden_size=128,
        num_hidden_layers=1,
        num_labels=args.num_classes
    )

    model = BertForSequenceClassification(config)
    model.to(args.device)

    params = [{
        "params": [p for n, p in model.named_parameters()],
        "weight_decay": args.weight_decay,
    }]

    optimizer = torch.optim.Adam(params, lr=args.learning_rate, eps=args.eps)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(args.num_epochs):
        train(train_loader, model, optimizer, criterion, args, epoch)
        validate(valid_loader, model, args, args.device, epoch)

    quantized_model = torch.quantization.quantize_dynamic(
        model.to('cpu'), {torch.nn.Linear}, dtype=torch.qint8
    )

    model = model.to('cpu')

    if args.verbose:
        print(quantized_model)

    print_size_of_model(model)
    print_size_of_model(quantized_model)

    time_evaluation(valid_loader, model, args, device='cpu')
    time_evaluation(valid_loader, quantized_model, args, device='cpu')


def main():
    params = initialize_parameters()
    run(params)


if __name__ == "__main__":
    main()
