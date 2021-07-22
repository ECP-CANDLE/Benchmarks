import logging
import os
import sys

import numpy as np
import torch
from sklearn.linear_model import LinearRegression
from torchvision.utils import save_image

from model import GeneralVae, PictureDecoder, PictureEncoder

logger = logging.getLogger('cairosvg')
logger.setLevel(logging.CRITICAL)

file_path = os.path.dirname(os.path.realpath(__file__))
lib_path = os.path.abspath(os.path.join(file_path, '..', '..', 'common'))
sys.path.append(lib_path)

import candle

additional_definitions = [
    {'name': 'batch_size', 'default': 64, 'type': int,
     'help': 'mini-batch size per process (default: 256)'},
    {'name': 'output_dir', 'help': 'output files path',
     'default': 'samples/'},
    {'name': 'checkpoint', 'type': str,
     'help': 'saved model to sample from'},
    {'name': 'num_samples', 'type': int, 'default': 64, 'help': 'number of samples to draw'},
    {'name': 'image', 'type': candle.str2bool, 'help': 'save images instead of numpy array'}
]

required = ['checkpoint']


class BenchmarkSample(candle.Benchmark):

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
    sampleBmk = BenchmarkSample(file_path, default_model, 'pytorch',
                                prog='sample_baseline',
                                desc='PyTorch ImageNet')

    # Initialize parameters
    gParameters = candle.finalize_parameters(sampleBmk)
    # logger.info('Params: {}'.format(gParameters))

    return gParameters


if __name__ == '__main__':
    gParams = initialize_parameters()
    args = candle.ArgumentStruct(**gParams)

#    args = get_args()

    starting_epoch = 1
    total_epochs = None

    # seed = 42
    # torch.manual_seed(seed)

    log_interval = 25
    LR = 5.0e-4

    cuda = True
    device = torch.device("cuda" if cuda and torch.cuda.is_available() else "cpu")

    encoder = PictureEncoder(rep_size=512)
    decoder = PictureDecoder(rep_size=512)

    checkpoint = None
    if args.checkpoint is not None:
        checkpoint = torch.load(args.model_path + '/' + args.checkpoint, map_location='cpu')
        print(f"Loading Checkpoint ({args.checkpoint}).")
        starting_epoch = checkpoint['epoch'] + 1
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        decoder.load_state_dict(checkpoint['decoder_state_dict'])

    encoder = encoder.to(device)
    decoder = decoder.to(device)
    model = GeneralVae(encoder, decoder, rep_size=512).to(device)

    def interpolate_points(x, y, sampling):
        ln = LinearRegression()
        data = np.stack((x, y))
        data_train = np.array([0, 1]).reshape(-1, 1)
        ln.fit(data_train, data)

        return ln.predict(sampling.reshape(-1, 1)).astype(np.float32)

    times = int(args.num_samples / args.batch_size)
    print(
        f"Using batch size {args.batch_size} and sampling {times} times for a total of {args.batch_size * times} samples drawn. Saving {'images' if args.image else 'numpy array'}")
    samples = []
    for i in range(times):
        with torch.no_grad():
            sample = torch.randn(args.batch_size, 512).to(device)
            sample = model.decode(sample).cpu()

            if args.image:
                save_image(sample.view(args.batch_size, 3, 256, 256),
                           args.output_dir + '/sample_' + str(i) + '.png')
            else:
                samples.append(sample.view(args.batch_size, 3, 256, 256).cpu().numpy())

    if not args.image:
        samples = np.concatenate(samples, axis=0)
        np.save(f"{args.output_dir}/samples.npy", samples)
