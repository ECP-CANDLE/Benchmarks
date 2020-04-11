import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from loguru import logger

import example_setup as bmk
import darts
import candle

from operations import (
    Stem, OPS, PRIMITIVES
)


def initialize_parameters():
    """ Initialize the parameters for the Uno example """

    uno_example = bmk.UnoExample(
        bmk.file_path,
        'default_model.txt',
        'pytorch',
        prog='uno_example',
        desc='Differentiable Architecture Search - Uno example',
    )

    # Initialize parameters
    gParameters = candle.finalize_parameters(uno_example)
    return gParameters


def run(params):
    args = candle.ArgumentStruct(**params)

    args.cuda = torch.cuda.is_available()
    device = torch.device(f"cuda" if args.cuda else "cpu")
    darts.banner(device=device)

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            '../data', train=True, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])),
            batch_size=args.batch_size, shuffle=True)

    valid_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            '../data', train=Fale, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])),
            batch_size=args.batch_size, shuffle=True)

    tasks = {
        'digits': 10,
    }

    criterion = nn.CrossEntropyLoss().to(device)

    stem = Stem(cell_dim=100)

    model = darts.Network(
        stem, cell_dim=100, primitives=PRIMITIVES, ops=OPS,
        tasks=tasks, criterion=criterion, device=device
    ).to(device)

    architecture = darts.Architecture(model, args, device=device)

    optimizer = optim.SGD(
        model.parameters(),
        args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        float(args.epochs),
        eta_min=args.lr_min
    )

    train_meter = darts.EpochMeter(tasks, 'train')
    valid_meter = darts.EpochMeter(tasks, 'valid')

    for epoch in range(args.epochs):

        scheduler.step()
        lr = scheduler.get_lr()[0]
        logger.info(f'\nEpoch: {epoch} lr: {lr}')

        genotype = model.genotype()
        logger.info(f'Genotype: {genotype}\n')

        train(
            trainloader,
            model,
            architecture,
            criterion,
            optimizer,
            lr,
            args,
            tasks,
            train_meter,
            device
        )

        validate(validloader, model, criterion, args, tasks, valid_meter, device)


def train(trainloader,
          model,
          architecture,
          criterion,
          optimizer,
          lr,
          args,
          tasks,
          meter,
          device):

    valid_iter = iter(trainloader)

    for step, (data, target) in enumerate(trainloader):

        batch_size = data.size(0)
        model.train()

        data = darts.to_device(data, device)
        target = darts.to_device(target, device)

        x_search, target_search = next(valid_iter)
        x_search = darts.to_device(x_search, device)
        target_search = darts.to_device(target_search, device)

        # 1. update alpha
        architecture.step(
            data,
            target,
            x_search,
            target_search,
            lr,
            optimizer,
            unrolled=args.unrolled
        )

        logits = model(data)
        loss = darts.multitask_loss(target, logits, criterion, reduce='mean')

        # 2. update weight
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        prec1 = darts.multitask_accuracy_topk(logits, target, topk=(1,))
        meter.update_batch_loss(loss.item(), batch_size)
        meter.update_batch_accuracy(prec1, batch_size)

        if step % args.log_interval == 0:
            logger.info(f'Step: {step} loss: {meter.loss_meter.avg:.4}')

    meter.update_epoch()
    meter.save(args.savepath)


def validate(validloader, model, criterion, args, tasks, meter, device):
    model.eval()
    with torch.no_grad():
        for step, (data, target) in enumerate(validloader):

            data = darts.to_device(data, device)
            target = darts.to_device(target, device)

            batch_size = data.size(0)

            logits = model(data)
            loss = darts.multitask_loss(target, logits, criterion, reduce='mean')

            prec1 = darts.multitask_accuracy_topk(logits, target, topk=(1,))
            meter.update_batch_loss(loss.item(), batch_size)
            meter.update_batch_accuracy(prec1, batch_size)

            if step % args.log_interval == 0:
                logger.info(f'>> Validation: {step} loss: {meter.loss_meter.avg:.4}')

    meter.update_epoch()
    meter.save(args.savepath)


def main():
    params = initialize_parameters()
    run(params)


if __name__=='__main__':
    main()
