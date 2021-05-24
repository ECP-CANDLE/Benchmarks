import os
import sys

import torch
from torchvision.utils import save_image
from tqdm import tqdm

from vqvae import VQVAE
from pixelsnail import PixelSNAIL

file_path = os.path.dirname(os.path.realpath(__file__))
lib_path = os.path.abspath(os.path.join(file_path, '..'))
sys.path.append(lib_path)
lib_path2 = os.path.abspath(os.path.join(file_path, '..', '..', 'common'))
sys.path.append(lib_path2)


import candle

additional_definitions = [
    {'name': 'vqvae',
        'type': str,
        'default': 'histology.pt',
        'help': ''},
    {'name': 'top',
        'type': str,
        'default': 'top.pt',
        'help': ''},
    {'name': 'bottom',
        'type': str,
        'default': 'bottom.pt',
        'help': ''},
    {'name': 'temp',
        'type': float,
        'default': 1.0,
        'help': ''},
    {'name': 'filename',
        'type': str,
        'default': '',
        'help': ''},
    {'name': 'use_gpus',
        'type': candle.str2bool,
        'default': False,
        'help': ''},
]

required = [
    'top',
    'bottom',
    'vqvae',
    'filename',
    'batch_size',
]


class SampleBk(candle.Benchmark):

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


def initialize_parameters(default_model='sample_default_model.txt'):

    # Build benchmark object
    sample = SampleBk(file_path, default_model, 'pytorch',
                      prog='sample_baseline', desc='Histology Sample - Examples')

    print("Created sample benchmark")

    # Initialize parameters
    gParameters = candle.finalize_parameters(sample)
    print("Parameters initialized")

    return gParameters


@torch.no_grad()
def sample_model(model, device, batch, size, temperature, condition=None):
    row = torch.zeros(batch, *size, dtype=torch.int64).to(device)
    cache = {}

    for i in tqdm(range(size[0])):
        for j in range(size[1]):
            out, cache = model(row[:, : i + 1, :], condition=condition, cache=cache)
            prob = torch.softmax(out[:, :, i, j] / temperature, 1)
            sample = torch.multinomial(prob, 1).squeeze(-1)
            row[:, i, j] = sample

    return row


def get_data(gParams):
    data_url = gParams['data_url']
    gParams['vqvae'] = candle.fetch_file(data_url + gParams['vqvae'], subdir='Examples/histogen')
    gParams['top'] = candle.fetch_file(data_url + gParams['top'], subdir='Examples/histogen')
    gParams['bottom'] = candle.fetch_file(data_url + gParams['bottom'], subdir='Examples/histogen')


def load_model(model, checkpoint, device):
    ndevices = torch.cuda.device_count()
    if ndevices == 0:
        ckpt = torch.load(os.path.join('checkpoint', checkpoint), map_location=torch.device('cpu'))
    else:
        ckpt = torch.load(os.path.join('checkpoint', checkpoint))

    if 'args' in ckpt:
        args = ckpt['args']

    if model == 'vqvae':
        model = VQVAE()

    elif model == 'pixelsnail_top':
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

    elif model == 'pixelsnail_bottom':
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
        ckpt = ckpt['model']

    model.load_state_dict(ckpt)
    model = model.to(device)
    model.eval()

    return model


def run(params):

    # this needs to go first to overwrite data paths
    get_data(params)

    args = candle.ArgumentStruct(**params)

    # Configure GPUs
    if args.use_gpus:
        device_ids = []
        ndevices = torch.cuda.device_count()
        if ndevices > 1:
            for i in range(ndevices):
                device_i = torch.device('cuda:' + str(i))
                device_ids.append(device_i)
            device = device_ids[0]
        elif ndevices == 1:
            device = torch.device('cuda:0')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device('cpu')

    model_vqvae = load_model('vqvae', args.vqvae, device)
    model_top = load_model('pixelsnail_top', args.top, device)
    model_bottom = load_model('pixelsnail_bottom', args.bottom, device)

    top_sample = sample_model(model_top, device, args.batch_size, [32, 32], args.temp)
    bottom_sample = sample_model(
        model_bottom, device, args.batch_size, [64, 64], args.temp, condition=top_sample
    )

    decoded_sample = model_vqvae.decode_code(top_sample, bottom_sample)
    decoded_sample = decoded_sample.clamp(-1, 1)

    save_image(decoded_sample, args.filename, normalize=True, range=(-1, 1))


def main():
    params = initialize_parameters()
    run(params)


if __name__ == '__main__':
    main()
