import sys
import os

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from argparse import SUPPRESS

try:
    from apex import amp

except ImportError:
    amp = None

from dataset import LMDBDataset
from pixelsnail import PixelSNAIL
from scheduler import CycleScheduler

file_path = os.path.dirname(os.path.realpath(__file__))
lib_path = os.path.abspath(os.path.join(file_path, '..'))
sys.path.append(lib_path)
lib_path2 = os.path.abspath(os.path.join(file_path, '..', '..', 'common'))
sys.path.append(lib_path2)


import candle

additional_definitions = [
    {'name': 'sched_mode',
        'type': str,
        'default': None,
        'help': 'Mode of learning rate scheduler'},
    {'name': 'lmdb_filename',
        'type': str,
        'default': SUPPRESS,
        'help': 'lmdb dataset path'},
    {'name': 'amp',
        'type': str,
        'default': 'O0',
        'help': ''},
    {'name': 'hier',
        'type': str,
        'default': 'top',
        'help': ''},
    {'name': 'channel',
        'type': int,
        'default': 256,
        'help': ''},
    {'name': 'n_res_block',
        'type': int,
        'default': 4,
        'help': ''},
    {'name': 'n_res_channel',
        'type': int,
        'default': 256,
        'help': ''},
    {'name': 'n_out_res_block',
        'type': int,
        'default': 0,
        'help': ''},
    {'name': 'n_cond_res_block',
        'type': int,
        'default': 3,
        'help': ''},
    {'name': 'ckpt_restart',
        'type': str,
        'default': None,
        'help': 'Checkpoint to restart from'},
]

required = [
    'batch_size',
    'epochs',
    'hier',
    'learning_rate',
    'channel',
    'n_res_block',
    'n_res_channel',
    'n_out_res_block',
    'n_cond_res_block',
    'dropout',
    'amp',
    'sched_mode',
    'lmdb_filename',
]


class TrPxSnBk(candle.Benchmark):

    def set_locals(self):
        """Functionality to set variables specific for the benchmark
        - required: set of required parameters for the benchmark.
        - additional_definitions: list of dictionaries describing the additional parameters for the
        benchmark.
        """

        if required is not None:
            self.required = set(required)
        if additional_definitions is not None:
            self.additional_definitions = additional_definitions


def initialize_parameters(default_model='train_pixelsnail_default_model.txt'):

    # Build benchmark object
    trpsn = TrPxSnBk(file_path, default_model, 'pytorch',
                     prog='train_pixelsnail_baseline',
                     desc='Histology train pixelsnail - Examples')

    print("Created sample benchmark")

    # Initialize parameters
    gParameters = candle.finalize_parameters(trpsn)
    print("Parameters initialized")

    return gParameters


def train(args, epoch, loader, model, optimizer, scheduler, device):
    loader = tqdm(loader)

    criterion = nn.CrossEntropyLoss()

    for i, (top, bottom, label) in enumerate(loader):
        model.zero_grad()

        top = top.to(device)

        if args.hier == 'top':
            target = top
            out, _ = model(top)

        elif args.hier == 'bottom':
            bottom = bottom.to(device)
            target = bottom
            out, _ = model(bottom, condition=top)

        loss = criterion(out, target)
        loss.backward()

        if scheduler is not None:
            scheduler.step()
        optimizer.step()

        _, pred = out.max(1)
        correct = (pred == target).float()
        accuracy = correct.sum() / target.numel()

        lr = optimizer.param_groups[0]['lr']

        loader.set_description(
            (
                f'epoch: {epoch + 1}; loss: {loss.item():.5f}; '
                f'acc: {accuracy:.5f}; lr: {lr:.5f}'
            )
        )


class PixelTransform:
    def __init__(self):
        pass

    def __call__(self, input):
        ar = np.array(input)

        return torch.from_numpy(ar).long()


def run(params):

    args = candle.ArgumentStruct(**params)
    # Configure GPUs
    ndevices = torch.cuda.device_count()
    if ndevices < 1:
        raise Exception('No CUDA gpus available')

    device = 'cuda'

    dataset = LMDBDataset(args.lmdb_filename)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True
    )

    ckpt = {}

    if args.ckpt_restart is not None:
        ckpt = torch.load(args.ckpt_restart)
        args = ckpt['args']

    if args.hier == 'top':
        model = PixelSNAIL(
            [32, 32],
            512,
            args.channel,
            5,
            4,
            args.n_res_block,
            args.n_res_channel,
            dropout=args.dropout,
            n_out_res_block=args.n_out_res_block,
        )

    elif args.hier == 'bottom':
        model = PixelSNAIL(
            [64, 64],
            512,
            args.channel,
            5,
            4,
            args.n_res_block,
            args.n_res_channel,
            attention=False,
            dropout=args.dropout,
            n_cond_res_block=args.n_cond_res_block,
            cond_res_channel=args.n_res_channel,
        )

    if 'model' in ckpt:
        model.load_state_dict(ckpt['model'])

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    if amp is not None:
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.amp)

    model = nn.DataParallel(model)
    model = model.to(device)

    scheduler = None
    if args.sched_mode == 'cycle':
        scheduler = CycleScheduler(
            optimizer, args.learning_rate, n_iter=len(loader) * args.epochs, momentum=None
        )

    for i in range(args.epochs):
        train(args, i, loader, model, optimizer, scheduler, device)
        torch.save(
            {'model': model.module.state_dict(), 'args': args},
            f'{args.ckpt_directory}/checkpoint/pixelsnail_{args.hier}_{str(i + 1).zfill(3)}.pt',
        )


def main():
    params = initialize_parameters()
    run(params)


if __name__ == '__main__':
    main()
