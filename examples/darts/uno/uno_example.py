import logging

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

import example_setup as bmk
import darts
import candle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("darts_uno")


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
    device = torch.device("cuda" if args.cuda else "cpu")
    darts.banner(device=device)

    train_data = darts.Uno('./data', 'train', download=True)
    valid_data = darts.Uno('./data', 'test')

    trainloader = DataLoader(train_data, batch_size=args.batch_size)
    validloader = DataLoader(valid_data, batch_size=args.batch_size)

    criterion = nn.CrossEntropyLoss().to(device)

    tasks = {
        'response': 2,
    }

    model = darts.LinearNetwork(
        input_dim=942, tasks=tasks, criterion=criterion, device=device
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

    genotype_store = darts.GenotypeStorage(root=args.save_path)

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
            genotype,
            genotype_store,
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
          genotype,
          genotype_store,
          device):

    valid_iter = iter(trainloader)
    min_accuracy = 0.0
    for step, (data, target) in enumerate(trainloader):

        batch_size = data.size(0)
        model.train()

        data = darts.to_device(data, device)
        target = darts.to_device(target, device)

        x_search, target_search = next(valid_iter)
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

        accuracy_avg = meter.acc_meter.get_avg_accuracy('response')
        if accuracy_avg > min_accuracy:
            genotype_store.save_genotype(genotype)
            min_accuracy = accuracy_avg

        if step % args.log_interval == 0:
            logger.info(f'Step: {step} loss: {meter.loss_meter.avg:.4}')

    meter.update_epoch()
    meter.save(args.save_path)


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
    meter.save(args.save_path)


def main():
    params = initialize_parameters()
    run(params)


if __name__ == '__main__':
    main()
