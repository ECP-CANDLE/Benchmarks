import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

import darts
import candle
import example_setup as bmk

from uno_darts import train, validate


def initialize_parameters():
    """ Initialize the parameters for the Uno example """

    uno_example = bmk.UnoExample(
        bmk.file_path,
        'uno_default_model.txt',
        'pytorch',
        prog='uno_example',
        desc='Differentiable Architecture Search - Uno example',
    )

    # Initialize parameters
    gParameters = candle.finalize_parameters(p3b5_bench)
    return gParameters


def run(params):
    args = candle.ArgumentStruct(**params)

    args.cuda = torch.cuda.is_available()
    device = torch.device(f"cuda" if args.cuda else "cpu")
    darts.banner(device=device)

    train_data = darts.Uno(args.datapath, 'train', download=True)
    valid_data = darts.Uno(args.datapath, 'test')

    train_data = sample(train_data, len(valid_data))

    trainloader = DataLoader(train_data, batch_size=args.batch_size)
    validloader = DataLoader(valid_data, batch_size=args.batch_size)

    criterion = nn.CrossEntropyLoss().to(device)

    tasks = {
        'response': 2,
    }

    model = darts.LinearNetwork(
        tasks=tasks, criterion=criterion, device=device
    ).to(device)

    architecture = darts.Architecture(model, args, device=device)

    optimizer = optim.SGD(
        model.parameters(),
        args.lr,
        momentum=args.momentum,
        weight_decay=args.wd
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
          validloader,
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

        data = to_device(data, device)
        target = to_device(target, device)

        x_search, target_search = next(valid_iter)
        x_search = to_device(x_search, device)
        target_search = to_device(target_search, device)

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
        loss = multitask_loss(target, logits, criterion, reduce='mean')

        # 2. update weight
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        prec1 = multitask_accuracy_topk(logits, target, topk=(1,))
        meters.update_batch_loss(loss.item(), batch_size)
        meters.update_batch_accuracy(prec1, batch_size)

        if step % args.log_interval == 0:
            logger.info(f'Step: {step} loss: {meters.loss_meter.avg:.4}')

    meters.update_epoch()
    meters.save(args.results_path)


def validate(validloader, model, criterion, args, tasks, meters, device):
    model.eval()
    with torch.no_grad():
        for step, (data, target) in enumerate(validloader):

            data = to_device(data, device)
            target = to_device(target, device)

            batch_size = data.size(0)

            logits = model(data)
            loss = multitask_loss(target, logits, criterion, reduce='mean')

            prec1 = multitask_accuracy_topk(logits, target, topk=(1,))
            meters.update_batch_loss(loss.item(), batch_size)
            meters.update_batch_accuracy(prec1, batch_size)

            if step % args.log_interval == 0:
                logger.info(f'>> Validation: {step} loss: {meters.loss_meter.avg:.4}')

    meters.update_epoch()
    meters.save(args.results_path)


def main():
    params = initialize_parameters()
    run(params)


if __name__=='__main__':
    main()
