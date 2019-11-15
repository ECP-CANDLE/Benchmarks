import os
import sys
import argparse
import candle
import p3b5 as bmk

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from darts.api.config import banner
from darts.modules.network import Network
from darts.architecture import Architecture
from darts.meters.average import AverageMeter
from darts.functional import multitask_loss, multitask_accuracy
from darts.meters.accuracy import MultitaskAccuracyMeter
from darts.utils.logging import log_accuracy


file_path = os.path.dirname(os.path.realpath(__file__))
lib_path = os.path.abspath(os.path.join(file_path, '..'))
sys.path.append(lib_path)
lib_path2 = os.path.abspath(os.path.join(file_path, '..', '..', 'common'))
sys.path.append(lib_path2)


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
        loss = multitask_loss(target, logits, criterion, reduce='mean')

        # 2. update weight
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        prec1 = multitask_accuracy(target, logits)
        losses.update(loss.item(), batch_size)
        top1.update(prec1, batch_size)

        if step % args.log_interval == 0:
            print(f'Step: {step} loss: {losses.avg:.4}')
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
            loss = multitask_loss(target, logits, criterion, reduce='mean')

            prec1 = multitask_accuracy(target, logits)
            losses.update(loss.item(), batch_size)
            top1.update(prec1, batch_size)

            if step % args.log_interval == 0:
                print(f'>> Validation: {step} loss: {losses.avg:.4}')
                log_accuracy(top1, 'valid')

    return top1, losses.avg


if __name__=='__main__':
    main()
