import torch
import p3b7 as bmk
import candle

import numpy as np

import torch.nn as nn
from torch.utils.data import DataLoader

from data import P3B3, Egress
from mtcnn import MTCNN, Hparams
from util import to_device
from meters import AccuracyMeter
from metrics import F1Meter

from prune import (
    negative_prune, min_max_prune,
    create_prune_masks, remove_prune_masks
)


TASKS = {
    'subsite': 15,
    'laterality': 3,
    'behavior': 3,
    'grade': 3,
}

TRAIN_F1_MICRO = F1Meter(TASKS, 'micro')
VALID_F1_MICRO = F1Meter(TASKS, 'micro')

TRAIN_F1_MACRO = F1Meter(TASKS, 'macro')
VALID_F1_MACRO = F1Meter(TASKS, 'macro')


def initialize_parameters():
    """Initialize the parameters for the P3B7 benchmark """
    p3b7_bench = bmk.BenchmarkP3B7(
        bmk.file_path,
        "default_model.txt",
        "pytorch",
        prog="p3b7",
        desc="Network pruning",
    )

    gParameters = candle.finalize_parameters(p3b7_bench)
    return gParameters


def fetch_data(gParameters):
    """Download and unpack data

    Args:
        gParameters: parameters from candle

    Returns:
        path to where the data is located
    """
    path = gParameters.data_url
    fpath = candle.fetch_file(
        path + gParameters.train_data, 'Pilot3', unpack=True
    )
    return fpath


def get_synthetic_data(args):
    """Initialize data loaders

    Args:
        datapath: path to the synthetic data

    Returns:
        train and valid data
    """
    datapath = fetch_data(args)
    train_data = P3B3(datapath, 'train')
    valid_data = P3B3(datapath, 'test')
    return train_data, valid_data


def get_egress_data(tasks):
    """Initialize egress tokenized data loaders

    Args:
        args: CANDLE ArgumentStruct
        tasks: dictionary of the number of classes for each task

    Returns:
        train and valid data
    """
    train_data = Egress('./data', 'train')
    valid_data = Egress('./data', 'valid')
    return train_data, valid_data


def train(model, loader, optimizer, device, epoch):
    accmeter = AccuracyMeter(TASKS, loader)

    total_loss = 0
    for idx, (data, target) in enumerate(loader):
        optimizer.zero_grad()
        data, target = data.to(device), to_device(target, device)
        logits = model(data)
        _ = TRAIN_F1_MICRO.f1(to_device(logits, 'cpu'), to_device(target, 'cpu'))
        _ = TRAIN_F1_MACRO.f1(to_device(logits, 'cpu'), to_device(target, 'cpu'))
        loss = model.loss_value(logits, target, reduce="mean")
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        accmeter.update(logits, target)

    avg_loss = total_loss / len(loader.dataset)

    accmeter.update_accuracy()
    print(f'\nEpoch {epoch} Training Accuracy:')
    accmeter.print_task_accuracies()
    accmeter.reset()
    return avg_loss


def evaluate(model, loader, device):
    accmeter = AccuracyMeter(TASKS, loader)

    loss = 0
    model.eval()
    with torch.no_grad():
        for idx, (data, target) in enumerate(loader):
            data, target = data.to(device), to_device(target, device)
            logits = model(data)
            _ = VALID_F1_MICRO.f1(to_device(logits, 'cpu'), to_device(target, 'cpu'))
            _ = VALID_F1_MACRO.f1(to_device(logits, 'cpu'), to_device(target, 'cpu'))
            loss += model.loss_value(logits, target, reduce="mean").item()
            accmeter.update(logits, target)

    accmeter.update_accuracy()

    print(f'Validation accuracy:')
    accmeter.print_task_accuracies()

    loss /= len(loader.dataset)

    return loss


def save_dataframe(metrics, filename):
    """Save F1 metrics"""
    df = pd.DataFrame(metrics, index=[0])
    path = Path(ARGS.savepath).joinpath(f'f1/{filename}.csv')
    df.to_csv(path, index=False)


def run(args):
    args = candle.ArgumentStruct(**args)
    args.cuda = torch.cuda.is_available()
    args.device = torch.device(f"cuda" if args.cuda else "cpu")

    if args.use_synthetic_data:
        train_data, valid_data = get_synthetic_data(args)

        hparams = Hparams(
            kernel1=args.kernel1,
            kernel2=args.kernel2,
            kernel3=args.kernel3,
            embed_dim=args.embed_dim,
            n_filters=args.n_filters,
        )
    else:
        train_data, valid_data = get_egress_data(tasks)

        hparams = Hparams(
            kernel1=args.kernel1,
            kernel2=args.kernel2,
            kernel3=args.kernel3,
            embed_dim=args.embed_dim,
            n_filters=args.n_filters,
            vocab_size=len(train_data.vocab)
        )

    train_loader = DataLoader(train_data, batch_size=args.batch_size)
    valid_loader = DataLoader(valid_data, batch_size=args.batch_size)

    model = MTCNN(TASKS, hparams).to(args.device)
    model = create_prune_masks(model)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, eps=args.eps
    )

    train_epoch_loss = []
    valid_epoch_loss = []
    for epoch in range(args.epochs):
        train_loss = train(model, train_loader, optimizer, args.device, epoch)
        valid_loss = evaluate(model, valid_loader, args.device)
        train_epoch_loss.append(train_loss)
        valid_epoch_loss.append(valid_loss)

    model = remove_prune_masks(model)


def main():
    params = initialize_parameters()
    run(params)


if __name__ == "__main__":
    main()
