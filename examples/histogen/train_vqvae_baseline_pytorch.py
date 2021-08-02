import sys
import os

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from torchvision import datasets, transforms, utils

from tqdm import tqdm
from argparse import SUPPRESS

from vqvae import VQVAE
from scheduler import CycleScheduler
import distributed as dist


file_path = os.path.dirname(os.path.realpath(__file__))
lib_path = os.path.abspath(os.path.join(file_path, '..'))
sys.path.append(lib_path)
lib_path2 = os.path.abspath(os.path.join(file_path, '..', '..', 'common'))
sys.path.append(lib_path2)

port = (2 ** 15 + 2 ** 14 + hash(os.getuid()
        if sys.platform != "win32" else 1) % 2 ** 14)

import candle

additional_definitions = [
    {'name': 'dist_url',
        'type': str,
        'default': 'tcp://127.0.0.1:{port}',
        'help': ''},
    {'name': 'sched_mode',
        'type': str,
        'default': None,
        'help': 'Mode of learning rate scheduler'},
    {'name': 'n_gpu_per_machine',
        'type': int,
        'default': 1,
        'help': 'Number of gpus to use per machine'},
    {'name': 'data_dir',
        'type': str,
        'default': SUPPRESS,
        'help': 'dataset path'},
    {'name': 'image_size',
        'type': int,
        'default': 256,
        'help': 'Image size to use'},
]

required = [
    'n_gpu_per_machine',
    'dist_url',
    'epochs',
    'learning_rate',
    'sched_mode',
    'image_size',
]


class TrainBk(candle.Benchmark):

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


def initialize_parameters(default_model='train_vqvae_default_model.txt'):

    # Build benchmark object
    trvq = TrainBk(file_path, default_model, 'pytorch',
                   prog='train_vqae_baseline',
                   desc='Histology train vqae - Examples')

    print("Created sample benchmark")

    # Initialize parameters
    gParameters = candle.finalize_parameters(trvq)
    print("Parameters initialized")

    return gParameters


def train(epoch, loader, model, optimizer, scheduler, device):
    if dist.is_primary():
        loader = tqdm(loader)

    criterion = nn.MSELoss()

    latent_loss_weight = 0.25
    sample_size = 25

    mse_sum = 0
    mse_n = 0

    for i, (img, label) in enumerate(loader):
        model.zero_grad()

        img = img.to(device)

        out, latent_loss = model(img)
        recon_loss = criterion(out, img)
        latent_loss = latent_loss.mean()
        loss = recon_loss + latent_loss_weight * latent_loss
        loss.backward()

        if scheduler is not None:
            scheduler.step()
        optimizer.step()

        part_mse_sum = recon_loss.item() * img.shape[0]
        part_mse_n = img.shape[0]
        comm = {"mse_sum": part_mse_sum, "mse_n": part_mse_n}
        comm = dist.all_gather(comm)

        for part in comm:
            mse_sum += part["mse_sum"]
            mse_n += part["mse_n"]

        if dist.is_primary():
            lr = optimizer.param_groups[0]["lr"]

            loader.set_description(
                (
                    f"epoch: {epoch + 1}; mse: {recon_loss.item():.5f}; "
                    f"latent: {latent_loss.item():.3f}; avg mse: {mse_sum / mse_n:.5f}; "
                    f"lr: {lr:.5f}"
                )
            )

            if i % 100 == 0:
                model.eval()

                sample = img[:sample_size]

                with torch.no_grad():
                    out, _ = model(sample)

                utils.save_image(
                    torch.cat([sample, out], 0),
                    f"sample/{str(epoch + 1).zfill(5)}_{str(i).zfill(5)}.png",
                    nrow=sample_size,
                    normalize=True,
                    range=(-1, 1),
                )

                model.train()


def config_and_train(args):
    # Configure GPUs
    ndevices = torch.cuda.device_count()
    if ndevices < 1:
        raise Exception('No CUDA gpus available')

    device = "cuda"

    args.distributed = dist.get_world_size() > 1

    transform = transforms.Compose(
        [
            transforms.Resize(args.image_size),
            transforms.CenterCrop(args.image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    dataset = datasets.ImageFolder(args.data_dir, transform=transform)
    sampler = dist.data_sampler(dataset, shuffle=True, distributed=args.distributed)
    loader = DataLoader(
        dataset, batch_size=args.batch_size // args.n_gpu_per_machine, sampler=sampler, num_workers=2
    )

    model = VQVAE().to(device)

    if args.distributed:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[dist.get_local_rank()],
            output_device=dist.get_local_rank(),
        )

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = None
    if args.sched_mode == "cycle":
        scheduler = CycleScheduler(
            optimizer,
            args.lr,
            n_iter=len(loader) * args.epochs,
            momentum=None,
            warmup_proportion=0.05,
        )

    for i in range(args.epochs):
        train(i, loader, model, optimizer, scheduler, device)

        if dist.is_primary():
            torch.save(model.state_dict(), f"{args.ckpt_directory}/checkpoint/vqvae_{str(i + 1).zfill(3)}.pt")


def fetch_data(params):
    data_url = params['data_url']
    if params['data_dir'] is None:
        params['data_dir'] = candle.fetch_file(data_url + params['train_data'], subdir='Examples/histogen')
    else:
        tempfile = candle.fetch_file(data_url + params['train_data'], subdir='Examples/histogen')
        params['data_dir'] = os.path.join(os.path.dirname(tempfile), params['data_dir'])


def run(params):

    fetch_data(params)
    args = candle.ArgumentStruct(**params)

    dist.launch(config_and_train, args.n_gpu_per_machine, 1, 0, args.dist_url, args=(args,))


def main():
    params = initialize_parameters()
    run(params)


if __name__ == '__main__':
    main()
