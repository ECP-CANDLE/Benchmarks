import datetime
import itertools
import logging

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LinearRegression
from torch import nn, optim
from torchvision.utils import save_image

from dataloader import MoleLoader
from model import GeneralVae, PictureDecoder, PictureEncoder, customLoss
from utils import AverageMeter
import os
import sys

file_path = os.path.dirname(os.path.realpath(__file__))
lib_path = os.path.abspath(os.path.join(file_path, '..', '..', 'common'))
sys.path.append(lib_path)

import candle

logger = logging.getLogger('cairosvg')
logger.setLevel(logging.CRITICAL)

additional_definitions = [
    {'name': 'workers', 'default': 16, 'type': int,
     'help': 'number of workers, default 16'},
    {'name': 'batch-size', 'default': 256, 'type': int,
     'help': 'mini-batch size per process (default: 256)'},
    {'name': 'grad-clip', 'default': 2.0, 'type': float,
     'help': 'grad-clip, defautt 2.0'},
    {'name': 'log_interval', 'default': 25, 'type': int,
     'help': 'logging interval, default 25'},
    {'name': 'model_path', 'help': 'model save path', 'default': 'models/'},
    {'name': 'output_dir', 'help': 'output files path', 'default': 'output/'},
    {'name': 'checkpoint', 'default': None, 'type': str}
]

required = ['batch_size']


class BenchmarkImageVAE(candle.Benchmark):

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


def initialize_parameters(default_model='image_vae_default_model.txt'):

    # Build benchmark object
    image_vaeBmk = BenchmarkImageVAE(file_path, default_model, 'pytorch',
                                     prog='image_vae_baseline',
                                     desc='PyTorch ImageNet Training')

    # Initialize parameters
    gParameters = candle.finalize_parameters(image_vaeBmk)
    # logger.info('Params: {}'.format(gParameters))

    return gParameters


def get_batch_size(epoch, args):
    return args.batch_size


def clip_gradient(optimizer, grad_clip=1.0):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.
    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def run(gParams):
    args = candle.ArgumentStruct(**gParams)

    data_url = gParams['data_url'] + '/'
    train_data = gParams['train_data']
    test_data = gParams['test_data']

    train_file = candle.fetch_file(data_url + train_data, subdir='Examples/image_vae')
    test_file = candle.fetch_file(data_url + test_data, subdir='Examples/image_vae')

    starting_epoch = 1
    total_epochs = gParams['epochs']

    rng_seed = 42
    torch.manual_seed(rng_seed)

    log_interval = gParams['log_interval']
    LR = gParams['learning_rate']

    output_dir = args.output_dir
    save_files = args.model_path

    data_para = False
    if torch.cuda.device_count() > 1:
        data_para = True
    cuda = True
    device = torch.device("cuda" if cuda and torch.cuda.is_available() else "cpu")
    kwargs = {'num_workers': args.workers, 'pin_memory': True} if cuda else {'num_workers': args.workers}

    print("\nloading data...")
    smiles_lookup_train = pd.read_csv(train_file)
    print(smiles_lookup_train.head())
    smiles_lookup_test = pd.read_csv(test_file)
    print(smiles_lookup_test.head())
    print("Done.\n")

    encoder = PictureEncoder(rep_size=512)
    decoder = PictureDecoder(rep_size=512)

    checkpoint = None
    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        print(f"Loading Checkpoint ({args.checkpoint}). Starting at epoch: {checkpoint['epoch'] + 1}.")
        starting_epoch = checkpoint['epoch'] + 1
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        decoder.load_state_dict(checkpoint['decoder_state_dict'])

    encoder = encoder.to(device)
    decoder = decoder.to(device)
    model = GeneralVae(encoder, decoder, rep_size=512).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LR)

    if checkpoint is not None:
        print("using optimizer past state")
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    for param_group in optimizer.param_groups:
        param_group['lr'] = LR
    print("LR: {}".format(LR))

    loss_picture = customLoss()

    if data_para and torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
        loss_picture = nn.DataParallel(loss_picture)

    val_losses = []
    train_losses = []

    train_data = MoleLoader(smiles_lookup_train)
    train_loader_food = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size, shuffle=True, drop_last=True,
        **kwargs)

    val_data = MoleLoader(smiles_lookup_test)
    val_loader_food = torch.utils.data.DataLoader(
        val_data,
        batch_size=args.batch_size, shuffle=True, drop_last=True,
        **kwargs)

    def train(epoch, args):
        print("Epoch {}: batch_size {}".format(epoch, get_batch_size(epoch, args)))
        model.train()
        loss_meter = AverageMeter()
        for batch_idx, (_, data, _) in enumerate(train_loader_food):
            data = data.float().cuda()
            optimizer.zero_grad()

            recon_batch, mu, logvar, _ = model(data)

            loss2 = loss_picture(recon_batch, data, mu, logvar, epoch)
            loss2 = torch.sum(loss2)
            loss_meter.update(loss2.item(), int(recon_batch.shape[0]))

            loss2.backward()

            clip_gradient(optimizer, grad_clip=args.grad_clip)
            optimizer.step()

            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} {}'.format(
                    epoch, batch_idx * len(data), len(train_loader_food.dataset),
                    100. * batch_idx / len(train_loader_food),
                    loss_meter.avg, datetime.datetime.now()))

        print('====> Epoch: {} Average loss: {:.4f}'.format(
            epoch, loss_meter.avg))
        return loss_meter.avg

    def interpolate_points(x, y, sampling):
        ln = LinearRegression()
        data = np.stack((x, y))
        data_train = np.array([0, 1]).reshape(-1, 1)
        ln.fit(data_train, data)

        return ln.predict(sampling.reshape(-1, 1)).astype(np.float32)

    def test(epoch, args):
        model.eval()
        losses = AverageMeter()
        test_loss = 0
        with torch.no_grad():
            for i, (_, data, _) in enumerate(val_loader_food):
                data = data.float().cuda()

                recon_batch, mu, logvar, _ = model(data)
                print('recon', recon_batch.shape, mu.shape, logvar.shape, data.shape)
                loss2 = loss_picture(recon_batch, data, mu, logvar, epoch)
                loss2 = torch.sum(loss2)
                losses.update(loss2.item(), int(data.shape[0]))
                test_loss += loss2.item()
                if i == 0:
                    ##
                    n = min(data.size(0), 8)
                    comparison = torch.cat([data[:n],
                                            recon_batch.view(get_batch_size(epoch, args), 3, 256, 256)[:n]])
                    save_image(comparison.cpu(),
                               output_dir + 'reconstruction_' + str(epoch) + '.png', nrow=n)

                    del recon_batch

                    n_image_gen = 10
                    images = []
                    n_samples_linspace = 20
                    print(data.shape)
                    if data_para:
                        data_latent = model.module.encode_latent_(data[:25, ...])
                    else:
                        data_latent = model.encode_latent_(data)
                    print(data_latent.shape)
                    print(data.shape)
                    for i in range(n_image_gen):
                        pt_1 = data_latent[i * 2, ...].cpu().numpy()
                        pt_2 = data_latent[i * 2 + 1, ...].cpu().numpy()
                        sample_vec = interpolate_points(pt_1, pt_2,
                                                        np.linspace(0, 1, num=n_samples_linspace, endpoint=True))
                        sample_vec = torch.from_numpy(sample_vec).to(device)
                        if data_para:
                            images.append(model.module.decode(sample_vec).cpu())
                        else:
                            images.append(model.decode(sample_vec).cpu())

                    save_image(torch.cat(images), output_dir + 'linspace_' + str(epoch) + '.png',
                               nrow=n_samples_linspace)

        test_loss /= len(val_loader_food.dataset)
        print('====> Test set loss: {:.4f}'.format(test_loss))
        print('loss', losses.avg)

        val_losses.append(test_loss)

    if total_epochs is None:
        trn_rng = itertools.count(start=starting_epoch)
    else:
        trn_rng = range(starting_epoch, total_epochs + 1)

    for epoch in trn_rng:
        for param_group in optimizer.param_groups:
            print("Current learning rate is: {}".format(param_group['lr']))

        loss = train(epoch, args)
        test(epoch, args)

        if data_para:
            torch.save({
                'epoch': epoch,
                'encoder_state_dict': model.module.encoder.state_dict(),
                'decoder_state_dict': model.module.decoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, save_files + '/epoch_' + str(epoch) + '.pt')
            torch.save(model.module, "model_inf.pt")
        else:
            torch.save({
                'epoch': epoch,
                'encoder_state_dict': model.encoder.state_dict(),
                'decoder_state_dict': model.decoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, save_files + '/epoch_' + str(epoch) + '.pt')
            torch.save(model, "model_inf.pt")

        with torch.no_grad():
            sample = torch.randn(64, 512).to(device)
            if data_para:
                sample = model.module.decode(sample).cpu()
            else:
                sample = model.decode(sample).cpu()
            save_image(sample.view(64, 3, 256, 256),
                       output_dir + 'sample_' + str(epoch) + '.png')

    return loss


if __name__ == '__main__':
    gParams = initialize_parameters()
    loss = run(gParams)
