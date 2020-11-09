import torch
import argparse
#import candle
#import p3b6 as bmk

import numpy as np

import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from sklearn.metrics import f1_score

from bert import HiBERT
from random_data import MimicDatasetSynthetic


def parse_args():
    parser = argparse.ArgumentParser(description='Bert Mimic Synth')
    parser.add_argument('--batch_size', type=int, default=10,
                        help='batch size')
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='Adam learning rate')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Adam learning rate')
    parser.add_argument('--eps', type=float, default=1e-7,
                        help='Adam epsilon')
    parser.add_argument('--num_train_samples', type=int, default=100,
                        help='Number of training samples')
    parser.add_argument('--num_valid_samples', type=int, default=100,
                        help='Number of valid samples')
    parser.add_argument('--num_test_samples', type=int, default=100,
                        help='Number of test samples')
    parser.add_argument('--num_classes', type=int, default=10,
                        help='Number of clases')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='weight decay')
    parser.add_argument('--device', type=str, default='cuda',
                        help='path to the model weights')
    parser.add_argument('--pretrained_weights_path', type=str, 
                        help='path to the model weights')

    return parser.parse_args()


def initialize_parameters():
    """ Initialize the parameters for the P3B5 benchmark """

    p3b5_bench = bmk.BenchmarkP3B5(
        bmk.file_path,
        "default_model.txt",
        "pytorch",
        prog="p3b6",
        desc="BERT bench",
    )

    gParameters = candle.finalize_parameters(p3b6)
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
        segment_ids = batch["seg_ids"].to(args.device)
        input_mask = batch["masks"].to(args.device)

        logits = model(input_ids, input_mask, segment_ids)# n_segs)
        label_ids = batch["label"].to(args.device)

        loss = criterion(
            logits.view(-1, args.num_classes), label_ids)

        loss.backward()
        optimizer.step()

        train_loss += loss.mean()

        # track training loss
        if (idx + 1) % 100 == 0:
            train_loss = torch.tensor(train_loss)
            print(f"epoch: {epoch}, batch: {idx}, loss: {train_loss}")


def validate(dataloader, model, args, epoch):
    model.eval()

    preds = []
    labels = []

    with torch.no_grad():
        for idx, batch in enumerate(dataloader):

            input_ids = batch["tokens"].to(args.device)
            segment_ids = batch["seg_ids"].to(args.device)
            input_mask = batch["masks"].to(args.device)

            logits = model(input_ids, input_mask, segment_ids)#, n_segs)
            logits = torch.nn.Sigmoid()(logits)

            logits = logits.view(-1, args.num_classes).cpu().data.numpy()
            preds.append(np.rint(logits))
            labels.append(batch["label"].data.numpy())

    preds = np.concatenate(preds, 0)
    labels = np.concatenate(labels, 0)

    preds = torch.tensor(preds)

    labels = torch.tensor(labels)

    valid_f1 = f1_score(labels.flatten(), preds.flatten())

    print(f"epoch: {epoch}, validation F1: {valid_f1}")


def run(args):
    #args = candle.ArgumentStruct(**params)
    #args.cuda = torch.cuda.is_available()
    #args.device = torch.device(f"cuda" if args.cuda else "cpu")

    train_loader, valid_loader, test_loader = create_data_loaders(args)

    model = model = HiBERT(args.pretrained_weights_path, args.num_classes)
    model.to(args.device)

    params = [
        {
            "params": [p for n, p in model.named_parameters()],
            "weight_decay": args.weight_decay,
        }
    ]

    optimizer = torch.optim.Adam(params, lr=args.learning_rate, eps=args.eps)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(args.num_epochs):
        train(train_loader, model, optimizer, criterion, args, epoch)
        validate(valid_loader, model, args, epoch)


def main():
    #params = initialize_parameters()
    # Temporarily use argparse
    params = parse_args()
    run(params)


if __name__ == "__main__":
    main()
