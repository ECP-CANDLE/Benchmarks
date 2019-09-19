import os
import sys
import argparse
from loguru import logger

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datastore.data import P3B3
from hammer.metrics import multitask_accuracy_topk
from hammer.meters.average import AverageMeter

from darts.api.config import banner
from darts.modules.network import Network
from darts.architecture import Architecture
from darts.functional import multitask_loss
from darts.meters.accuracy import MultitaskAccuracyMeter
from darts.utils.logging import log_accuracy


def parse_args():
    parser = argparse.ArgumentParser(description='P3B3 Darts Example')
    parser.add_argument('--batch_size', type=int, default=100, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--lr', type=float, default=0.025, 
                        help='init learning rate')
    parser.add_argument('--lr_min', type=float, default=0.001, 
                        help='min learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, 
                        help='momentum')
    parser.add_argument('--wd', type=float, default=3e-4, 
                        help='weight decay')
    parser.add_argument('--grad_clip', type=float, default=5, 
                        help='gradient clipping range')
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='cuda device id for torch.device')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--datapath', type=str, default='/Users/yngtodd/data',
                        help='path to the dataset')
    parser.add_argument('--unrolled', action='store_true', default=False, 
                        help='use one-step unrolled validation loss')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    device = torch.device(f'cuda:{args.gpu_id}' if args.cuda else "cpu")
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
        args.lr, 
        momentum=args.momentum, 
        weight_decay=args.wd
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        float(args.epochs), 
        eta_min=args.lr_min
    )

    for epoch in range(args.epochs):

        scheduler.step()
        lr = scheduler.get_lr()[0]
        logger.info(f'\nEpoch: {epoch} lr: {lr}')

        genotype = model.genotype()
        logger.info(f'Genotype: {genotype}')

        #logger.debug(F.softmax(model.alphas_normal, dim=-1))
        #logger.debug(F.softmax(model.alphas_reduce, dim=-1))

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

        #utils.save(model, os.path.join(args.exp_path, 'search.pt'))


def train(trainloader, validloader, model, architecture, criterion, optimizer, lr, args, tasks, device):
    losses = AverageMeter('LossMeter')
    top1 = MultitaskAccuracyMeter(tasks)

    valid_iter = iter(trainloader)

    for step, (data, target) in enumerate(trainloader):

        batch_size = data.size(0)
        model.train()

        data = data.to(device)

        for task, label in target.items():
            target[task] = target[task].to(device)

        x_search, target_search = next(valid_iter)
        x_search = x_search.to(device)
       
        for task, label in target_search.items():
            target_search[task] = target_search[task].to(device)

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
        loss = multitask_loss(logits, target, criterion, reduce='mean')

        # 2. update weight
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        prec1 = multitask_accuracy_topk(logits, target, topk=(1,))
        losses.update(loss.item(), batch_size)
        top1.update(prec1, batch_size)

        if step % args.log_interval == 0:
            logger.info(f'Step: {step} loss: {losses.avg:.4}')
            log_accuracy(top1)

    return top1, losses.avg


def infer(validloader, model, criterion, args, tasks, device):
    losses = AverageMeter('LossMeter')
    top1 = MultitaskAccuracyMeter(tasks) 

    model.eval()

    with torch.no_grad():
        for step, (data, target) in enumerate(validloader):

            data = data.to(device)
            for task, label in target.items():
                target[task] = target[task].to(device)

            batch_size = data.size(0)

            logits = model(data)
            loss = multitask_loss(logits, target, criterion, reduce='mean')

            prec1 = multitask_accuracy_topk(logits, target, topk=(1,))
            losses.update(loss.item(), batch_size)
            top1.update(prec1, batch_size)

            if step % args.log_interval == 0:
                logger.info(f'>> Validation: {step} loss: {losses.avg:.4}')
                log_accuracy(top1, 'valid')

    return top1, losses.avg


if __name__=='__main__':
    main()
