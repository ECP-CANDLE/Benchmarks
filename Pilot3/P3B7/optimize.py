import torch
import candle
import p3b7 as bmk

import numpy as np

import torch.nn as nn
from torch.utils.data import DataLoader

from data import P3B3, Egress
from mtcnn import MTCNN, Hparams
from util import to_device
from meters import AccuracyMeter

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

TASKS = {
    'site': 70,
    'subsite': 325,
    'laterality': 7,
    'histology': 575,
    'behaviour': 4,
    'grade': 9 
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
    """Download and untar data

    Args:
        gParameters: parameters from candle

    Returns:
        path to where the data is located
    """
    path = gParameters['data_url']
    fpath = candle.fetch_file(
        path + gParameters['train_data'], 'Pilot3', untar=True
    )
    return fpath


def get_synthetic_data(args):
    """Initialize data loaders

    Args:
        datapath: path to the synthetic data

    Returns:
        train, valid, test data loaders
    """
    datapath = fetch_data(args)
    train_data = P3B3(datapath, 'train')
    valid_data = P3B3(datapath, 'test')
    trainloader = DataLoader(train_data, batch_size=args.batch_size)
    validloader = DataLoader(valid_data, batch_size=args.batch_size)
    return trainloader, validloader


def get_egress_data(args, tasks):
    """Initialize egress tokenized data loaders

    Args:
        args: CANDLE ArgumentStruct
        tasks: dictionary of the number of classes for each task

    Returns:
        train, valid, test data loaders
    """
    train_data = Egress('./data', 'train')
    valid_data = Egress('./data', 'valid')
    train_loader = DataLoader(train_data, batch_size=args.batch_size)
    valid_loader = DataLoader(valid_data, batch_size=args.batch_size)
    return train_loader, valid_loader


def train(dataloader, model, optimizer, args, epoch):
    model.train()

    epoch_loss = 0.0
    for idx, (data, target) in enumerate(dataloader):
        optimizer.zero_grad()
        data, target = data.to(device), to_device(target, device)
        logits = model(data)
        _ = TRAIN_F1_MICRO.f1(to_device(logits, 'cpu'), to_device(target, 'cpu'))
        _ = TRAIN_F1_MACRO.f1(to_device(logits, 'cpu'), to_device(target, 'cpu'))
        loss = model.loss_value(logits, target, reduce="mean")
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    epoch_loss /= len(loader.dataset)
    print(f"epoch: {epoch}, loss: {epoch_loss}")


def validate(dataloader, model, args, epoch):
    model.eval()
    accmeter = AccuracyMeter(TASKS, loader)

    with torch.no_grad():
        for idx, (data, target) in enumerate(dataloader):
            data = data.to(device)
            target = to_device(target, device)
            logits = model(data)
            _ = VALID_F1_MICRO.f1(to_device(logits, 'cpu'), to_device(target, 'cpu'))
            _ = VALID_F1_MACRO.f1(to_device(logits, 'cpu'), to_device(target, 'cpu'))
            accmeter.update(logits, target)

    accmeter.update_accuracy()
    print(f'Rank: {RANK} Validation accuracy:')
    accmeter.print_task_accuracies()


def save_dataframe(metrics, filename):
    """Save F1 metrics"""
    df = pd.DataFrame(metrics, index=[0])
    path = Path(ARGS.savepath).joinpath(f'f1/{filename}.csv')
    df.to_csv(path, index=False)


def run(args):
    args = candle.ArgumentStruct(**params)
    args.cuda = torch.cuda.is_available()
    args.device = torch.device(f"cuda" if args.cuda else "cpu")

    if args.use_synthetic:
        train_loader, valid_loader = get_synthetic_data(args)
    else:
        train_loader, valid_loader = get_egress_data(tasks)

    hparams = Hparams(
        kernel1=args.kernel1,
        kernel2=args.kernel2,
        kernel3=args.kernel3,
        embed_dim=args.embed_dim,
        n_filters=args.n_filters,
    )

    model = MTCNN(TASKS, hparams)
    model = create_prune_masks(model)
    model.to(args.device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, eps=args.eps
    )

    for epoch in range(args.num_epochs):
        train(train_loader, model, optimizer, args, epoch)
        validate(valid_loader, model, args, epoch)

    model = remove_prune_masks(model)


def main():
    params = initialize_parameters()
    run(params)


if __name__ == "__main__":
    main()
