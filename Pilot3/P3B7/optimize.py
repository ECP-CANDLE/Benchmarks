import torch
import candle
import p3b7 as bmk

import numpy as np

import torch.nn as nn
from torch.utils.data import DataLoader

from data import P3B3
from mtcnn import MTCNN, Hparams
from util import to_device
from meters import AccuracyMeter

from prune import (
   create_prune_masks, remove_prune_masks
)


TASKS = {
    'subsite': 15,
    'laterality': 3,
    'behavior': 3,
    'grade': 3,
}


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


def create_data_loaders(datapath):
    """Initialize data loaders

    Args:
        datapath: path to the synthetic data 

    Returns:
        train, valid, test data loaders
    """
    train_data = P3B3(datapath, 'train')
    valid_data = P3B3(datapath, 'test')

    trainloader = DataLoader(train_data, batch_size=args.batch_size)
    validloader = DataLoader(valid_data, batch_size=args.batch_size)
    return trainloader, validloader


def train(dataloader, model, optimizer, criterion, args, epoch):
    model.train()

    epoch_loss = 0.0
    for idx, (data, target) in enumerate(dataloader):
        optimizer.zero_grad()
        data = data.to(device)
        target = to_device(target, device)
        logits = model(data)
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
            accmeter.update(logits, target)

    accmeter.update_accuracy()
    print(f'Rank: {RANK} Validation accuracy:')
    accmeter.print_task_accuracies()


def run(args):
    args = candle.ArgumentStruct(**params)
    args.cuda = torch.cuda.is_available()
    args.device = torch.device(f"cuda" if args.cuda else "cpu")

    datapath = fetch_data(args)
    train_loader, valid_loader = create_data_loaders(datapath)

    hparams = Hparams(
        kernel1   = args.kernel1,
        kernel2   = args.kernel2,
	kernel3   = args.kernel3, 
	embed_dim = args.embed_dim,
	n_filters = args.n_filters,
    )

    model = MTCNN(TASKS, hparams)
    model = create_prune_masks(model)
    model.to(args.device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, eps=args.eps
    )

    criterion = nn.CrossEntropyLoss().to(device)

    for epoch in range(args.num_epochs):
        train(train_loader, model, optimizer, criterion, args, epoch)
        validate(valid_loader, model, args, epoch)

    model = remove_prune_masks(model)


def main():
    params = initialize_parameters()
    run(params)


if __name__ == "__main__":
    main()

