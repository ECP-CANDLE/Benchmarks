import os
import sys
import pickle

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import lmdb
from tqdm import tqdm
from argparse import SUPPRESS

from dataset import ImageFileDataset, CodeRow
from vqvae import VQVAE

file_path = os.path.dirname(os.path.realpath(__file__))
lib_path = os.path.abspath(os.path.join(file_path, '..'))
sys.path.append(lib_path)
lib_path2 = os.path.abspath(os.path.join(file_path, '..', '..', 'common'))
sys.path.append(lib_path2)


import candle

additional_definitions = [
    {'name': 'size',
        'type': int,
        'default': 256,
        'help': 'Image size to use'},
    {'name': 'data_dir',
        'type': str,
        'default': SUPPRESS,
        'help': 'dataset path'},
    {'name': 'lmdb_filename',
        'type': str,
        'default': SUPPRESS,
        'help': 'lmdb filename'},
    {'name': 'ckpt_restart',
        'type': str,
        'default': None,
        'help': 'Checkpoint to restart from'},
]

required = [
    'size',
    'data_dir',
    'lmdb_filename',
]


class ExtractCodeBk(candle.Benchmark):

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


def initialize_parameters(default_model='extract_code_default_model.txt'):

    # Build benchmark object
    excd = ExtractCodeBk(file_path, default_model,
                         'pytorch', prog='extract_code_baseline',
                         desc='Histology Extract Code - Examples')

    print("Created sample benchmark")

    # Initialize parameters
    gParameters = candle.finalize_parameters(excd)
    print("Parameters initialized")

    return gParameters


def extract(lmdb_env, loader, model, device):
    index = 0

    with lmdb_env.begin(write=True) as txn:
        pbar = tqdm(loader)

        for img, _, filename in pbar:
            img = img.to(device)

            _, _, _, id_t, id_b = model.encode(img)
            id_t = id_t.detach().cpu().numpy()
            id_b = id_b.detach().cpu().numpy()

            for file, top, bottom in zip(filename, id_t, id_b):
                row = CodeRow(top=top, bottom=bottom, filename=file)
                txn.put(str(index).encode('utf-8'), pickle.dumps(row))
                index += 1
                pbar.set_description(f'inserted: {index}')

        txn.put('length'.encode('utf-8'), str(index).encode('utf-8'))


def run(params):

    args = candle.ArgumentStruct(**params)
    # Configure GPUs
    ndevices = torch.cuda.device_count()
    if ndevices < 1:
        raise Exception('No CUDA gpus available')

    device = 'cuda'

    transform = transforms.Compose(
        [
            transforms.Resize(args.size),
            transforms.CenterCrop(args.size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    dataset = ImageFileDataset(args.data_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = VQVAE()
    if args.ckpt_restart is not None:
        model.load_state_dict(torch.load(args.ckpt_restart))
    model = model.to(device)
    model.eval()

    map_size = 100 * 1024 * 1024 * 1024

    env = lmdb.open(args.lmdb_filename, map_size=map_size)

    extract(env, loader, model, device)


def main():
    params = initialize_parameters()
    run(params)


if __name__ == '__main__':
    main()
