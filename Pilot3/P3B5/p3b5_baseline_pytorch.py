import candle
import p3b5_darts as bmk

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from darts.api.config import banner
from darts.modules.network import Network
from darts.architecture import Architecture
from darts.functional import multitask_loss
from darts.meters.accuracy import MultitaskAccuracyMeter
from darts.utils.logging import log_accurac

from p3b5_darts import train, infer


def initialize_parameters():
    """ Initialize the parameters for the P3B5 benchmark """

    p3b5_bench = bmk.BenchmarkP3B3(
        bmk.file_path,
        'p3b5_default_model.txt',
        'pytorch',
        prog='p3b5_baseline',
        desc='Differentiable Architecture Search - Pilot 3 Benchmark 5',
    )

    # Initialize parameters
    gParameters = candle.initialize_parameters(p3b5_bench)
    #bmk.logger.info('Params: {}'.format(gParameters))
    return gParameters


def fetch_data(gParameters):
    """ Download and untar data

    Args:
        gParameters: parameters from candle

    Returns:
        path to where the data is located
    """
    path = gParameters['data_url']
    fpath = candle.fetch_file(path + gParameters['train_data'], 'Pilot3', untar=True)
    return fpath


def run(params):
    args = candle.ArgumentStruct(**params)
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    device = torch.device(f'cuda' if args.cuda else "cpu")
    banner(device=device)

    train_data = P3B3(args.datapath, 'train', download=True)
    valid_data = P3B3(args.datapath, 'test')

    trainloader = DataLoader(train_data, batch_size=args.batch_size)
    validloader = DataLoader(valid_data, batch_size=args.batch_size)

    criterion = nn.CrossEntropyLoss().to(device)

    tasks = {
        'subsite': 6,
        'laterality': 2,
        'behavior': 2,
        'grade': 3
    }

    model = Network(tasks=tasks, criterion=criterion, device=device).to(device)
    architecture = Architecture(model, args, device=device)

    optimizer = optim.SGD(
        model.parameters(), 
        args.learning_rate, 
        momentum=args.momentum, 
        weight_decay=args.weight_decay,
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        float(args.epochs), 
        eta_min=args.learning_rate_min,
    )

    for epoch in range(args.epochs):

        scheduler.step()
        lr = scheduler.get_lr()[0]
        logger.info(f'\nEpoch: {epoch} lr: {lr}')

        genotype = model.genotype()
        logger.info(f'Genotype: {genotype}')

        # training
        train_acc, train_obj = train(
            trainloader, 
            validloader, 
            model, 
            architecture, 
            criterion, 
            optimizer, 
            lr, 
            args, 
            tasks,
            device
        )
       
        # validation
        valid_acc, valid_obj = infer(validloader, model, criterion, args, tasks, device)

        logger.info(f'\nEpoch {epoch} stats:')
        log_accuracy(train_acc, 'train')
        log_accuracy(valid_acc, 'valid')


def main():
    params = initialize_parameters()
    run(params)


if __name__=='__main__':
    main()