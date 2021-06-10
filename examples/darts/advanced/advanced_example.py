import torch
import torch.nn as nn
from torch import optim
from torchvision import datasets, transforms

import logging

import example_setup as bmk
import darts
import candle

from operations import (
    Stem, OPS
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("darts_advanced")


def initialize_parameters():
    """ Initialize the parameters for the Advanced example """

    uno_example = bmk.AdvancedExample(
        bmk.file_path,
        'default_model.txt',
        'pytorch',
        prog='advanced_example',
        desc='Differentiable Architecture Search - Advanced example',
    )

    # Initialize parameters
    gParameters = candle.finalize_parameters(uno_example)
    return gParameters


def run(params):
    args = candle.ArgumentStruct(**params)

    args.cuda = torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")
    darts.banner(device=device)

    trainloader = torch.utils.data.DataLoader(
        datasets.MNIST(
            './data', train=True, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])),
        batch_size=args.batch_size, shuffle=True)

    validloader = torch.utils.data.DataLoader(
        datasets.MNIST(
            './data', train=False, download=True,
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
        stem, cell_dim=100, classifier_dim=676,
        ops=OPS, tasks=tasks, criterion=criterion, device=device
    ).to(device)

    architecture = darts.Architecture(model, args, device=device)

    optimizer = optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        float(args.epochs),
        eta_min=args.learning_rate_min
    )

    train_meter = darts.EpochMeter(tasks, 'train')
    valid_meter = darts.EpochMeter(tasks, 'valid')

    for epoch in range(args.epochs):

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
            scheduler,
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
          scheduler,
          args,
          tasks,
          meter,
          device):

    valid_iter = iter(trainloader)

    for step, (data, target) in enumerate(trainloader):
        batch_size = data.size(0)
        model.train()
        target = _wrap_target(target)
        data = darts.to_device(data, device)
        target = darts.to_device(target, device)

        x_search, target_search = next(valid_iter)
        target_search = _wrap_target(target_search)
        x_search = darts.to_device(x_search, device)
        target_search = darts.to_device(target_search, device)

        lr = scheduler.get_lr()[0]

        # 1. update alpha
        architecture.step(
            data,
            target,
            x_search,
            target_search,
            lr,
            optimizer,
            unrolled=False
        )

        logits = model(data)
        loss = darts.multitask_loss(target, logits, criterion, reduce='mean')

        # 2. update weight
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()

        prec1 = darts.multitask_accuracy_topk(logits, target, topk=(1,))
        meter.update_batch_loss(loss.item(), batch_size)
        meter.update_batch_accuracy(prec1, batch_size)

        if step % args.log_interval == 0:
            logger.info(f'Step: {step} loss: {meter.loss_meter.avg:.4}')

    meter.update_epoch()
    meter.save(args.save_path)


def validate(validloader, model, criterion, args, tasks, meter, device):
    model.eval()
    with torch.no_grad():
        for step, (data, target) in enumerate(validloader):
            target = _wrap_target(target)

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
    meter.save(args.save_path)


def _wrap_target(target):
    """ Wrap the MNIST target in a dictionary

    The multitask classifier of DARTS expects a
    dictionary of target tasks. Here we simply wrap
    MNIST's target in a dictionary.
    """
    return {'digits': target}


def main():
    params = initialize_parameters()
    run(params)


if __name__ == '__main__':
    main()
