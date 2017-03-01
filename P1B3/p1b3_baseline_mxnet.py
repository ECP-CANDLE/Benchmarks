from __future__ import division, print_function

import argparse
import logging

import numpy as np

import mxnet as mx
from mxnet.io import DataBatch, DataIter

# # For non-interactive plotting
# import matplotlib as mpl
# mpl.use('Agg')
# import matplotlib.pyplot as plt

import p1b3


# Model and Training parameters

# Seed for random generation
SEED = 2016
# Size of batch for training
BATCH_SIZE = 100
# Number of training epochs
NB_EPOCH = 20
# Number of data generator workers
NB_WORKER = 1

# Percentage of dropout used in training
DROP = 0.1
# Activation function (options: 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear')
ACTIVATION = 'relu'
LOSS = 'mse'
OPTIMIZER = 'sgd'

# Type of feature scaling (options: 'maxabs': to [-1,1]
#                                   'minmax': to [0,1]
#                                   None    : standard normalization
SCALING = 'std'
# Features to (randomly) sample from cell lines or drug descriptors
FEATURE_SUBSAMPLE = 500#0
# FEATURE_SUBSAMPLE = 0

# Number of units in fully connected (dense) layers
D1 = 1000
D2 = 500
D3 = 100
D4 = 50
DENSE_LAYERS = [D1, D2, D3, D4]

# Number of units per locally connected layer
C1 = 10, 10, 5       # nb_filter, filter_length, stride
C2 = 0, 0, 0         # disabled layer
# CONVOLUTION_LAYERS = list(C1 + C2)
CONVOLUTION_LAYERS = [0, 0, 0]
POOL = 10

MIN_LOGCONC = -5.
MAX_LOGCONC = -4.

CATEGORY_CUTOFFS = [0.]

np.set_printoptions(threshold=np.nan)
np.random.seed(SEED)


def get_parser():
    parser = argparse.ArgumentParser(prog='p1b3_baseline',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="increase output verbosity")
    parser.add_argument("-a", "--activation", action="store",
                        default=ACTIVATION,
                        help="keras activation function to use in inner layers: relu, tanh, sigmoid...")
    parser.add_argument("-b", "--batch_size", action="store",
                        default=BATCH_SIZE, type=int,
                        help="batch size")
    parser.add_argument("-c", "--convolution", action="store", nargs='+', type=int,
                        default=CONVOLUTION_LAYERS,
                        help="integer array describing convolution layers: conv1_nb_filter, conv1_filter_len, conv1_stride, conv2_nb_filter, conv2_filter_len, conv2_stride ...")
    parser.add_argument("-d", "--dense", action="store", nargs='+', type=int,
                        default=DENSE_LAYERS,
                        help="number of units in fully connected layers in an integer array")
    parser.add_argument("-e", "--epochs", action="store",
                        default=NB_EPOCH, type=int,
                        help="number of training epochs")
    parser.add_argument("-l", "--locally_connected", action="store_true",
                        default=False,  # TODO: not currently supported
                        help="use locally connected layers instead of convolution layers")
    parser.add_argument("-o", "--optimizer", action="store",
                        default=OPTIMIZER,
                        help="keras optimizer to use: sgd, rmsprop, ...")
    parser.add_argument("--drop", action="store",
                        default=DROP, type=float,
                        help="ratio of dropout used in fully connected layers")
    parser.add_argument("--loss", action="store",
                        default=LOSS,
                        help="keras loss function to use: mse, ...")
    parser.add_argument("--pool", action="store",
                        default=POOL, type=int,
                        help="pooling layer length")
    parser.add_argument("--scaling", action="store",
                        default=SCALING,
                        help="type of feature scaling; 'minabs': to [-1,1]; 'minmax': to [0,1], 'std': standard unit normalization; None: no normalization")
    parser.add_argument("--drug_features", action="store",
                        default="descriptors",
                        help="use dragon7 descriptors, latent representations from Aspuru-Guzik's SMILES autoencoder, or both, or random features; 'descriptors','latent', 'both', 'noise'")
    parser.add_argument("--feature_subsample", action="store",
                        default=FEATURE_SUBSAMPLE, type=int,
                        help="number of features to randomly sample from each category (cellline expression, drug descriptors, etc), 0 means using all features")
    parser.add_argument("--min_logconc", action="store",
                        default=MIN_LOGCONC, type=float,
                        help="min log concentration of dose response data to use: -3.0 to -7.0")
    parser.add_argument("--max_logconc", action="store",
                        default=MAX_LOGCONC, type=float,
                        help="max log concentration of dose response data to use: -3.0 to -7.0")
    parser.add_argument("--subsample", action="store",
                        default='naive_balancing',
                        help="dose response subsample strategy; None or 'naive_balancing'")
    parser.add_argument("--category_cutoffs", action="store", nargs='+', type=float,
                        default=CATEGORY_CUTOFFS,
                        help="list of growth cutoffs (between -1 and +1) seperating non-response and response categories")
    parser.add_argument("--train_samples", action="store",
                        default=0, type=int,
                        help="overrides the number of training samples if set to nonzero")
    parser.add_argument("--val_samples", action="store",
                        default=0, type=int,
                        help="overrides the number of validation samples if set to nonzero")
    parser.add_argument("--save", action="store",
                        default='save',
                        help="prefix of output files")
    parser.add_argument("--scramble", action="store_true",
                        help="randomly shuffle dose response data")
    parser.add_argument("--workers", action="store",
                        default=NB_WORKER, type=int,
                        help="number of data generator workers")
    parser.add_argument("--gpus", action="store", nargs='*',
                        default=[], type=int,
                        help="set IDs of GPUs to use")

    return parser


def extension_from_parameters(args):
    """Construct string for saving model with annotation of parameters"""
    ext = '.mx'
    ext += '.A={}'.format(args.activation)
    ext += '.B={}'.format(args.batch_size)
    ext += '.D={}'.format(args.drop)
    ext += '.E={}'.format(args.epochs)
    if args.feature_subsample:
        ext += '.F={}'.format(args.feature_subsample)
    if args.convolution:
        name = 'LC' if args.locally_connected else 'C'
        layer_list = list(range(0, len(args.convolution), 3))
        for l, i in enumerate(layer_list):
            nb_filter = args.convolution[i]
            filter_len = args.convolution[i+1]
            stride = args.convolution[i+2]
            if nb_filter <= 0 or filter_len <= 0 or stride <= 0:
                break
            ext += '.{}{}={},{},{}'.format(name, l+1, nb_filter, filter_len, stride)
        if args.pool and layer_list[0] and layer_list[1]:
            ext += '.P={}'.format(args.pool)
    for i, n in enumerate(args.dense):
        if n:
            ext += '.D{}={}'.format(i+1, n)
    ext += '.S={}'.format(args.scaling)

    return ext


class ConcatDataIter(DataIter):
    """Data iterator for concatenated features
    """

    def __init__(self, data_loader,
                 partition='train',
                 batch_size=32,
                 num_data=None,
                 shape=None):
        super(ConcatDataIter, self).__init__()
        self.data = data_loader
        self.batch_size = batch_size
        self.gen = p1b3.DataGenerator(data_loader, partition=partition, batch_size=batch_size, shape=shape, concat=True)
        self.num_data = num_data or self.gen.num_data
        self.cursor = 0
        self.gen = self.gen.flow()

    @property
    def provide_data(self):
        return [('concat_features', (self.batch_size, self.data.input_dim))]

    @property
    def provide_label(self):
        return [('growth', (self.batch_size,))]

    def reset(self):
        self.cursor = 0

    def iter_next(self):
        self.cursor += self.batch_size
        if self.cursor <= self.num_data:
            return True
        else:
            return False

    def next(self):
        if self.iter_next():
            x, y = next(self.gen)
            return DataBatch(data=[mx.nd.array(x)], label=[mx.nd.array(y)])
        else:
            raise StopIteration


def plot_network(net, filename):
    try:
        dot = mx.viz.plot_network(net)
    except ImportError:
        return
    try:
        dot.render(filename, view=False)
        print('Plotted network architecture in {}'.format(filename+'.pdf'))
    except Exception:
        return


def main():
    parser = get_parser()
    args = parser.parse_args()
    print('Args:', args)

    loggingLevel = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=loggingLevel, format='')

    ext = extension_from_parameters(args)

    loader = p1b3.DataLoader(feature_subsample=args.feature_subsample,
                             scaling=args.scaling,
                             drug_features=args.drug_features,
                             scramble=args.scramble,
                             min_logconc=args.min_logconc,
                             max_logconc=args.max_logconc,
                             subsample=args.subsample,
                             category_cutoffs=args.category_cutoffs)

    net = mx.sym.Variable('concat_features')
    out = mx.sym.Variable('growth')

    if args.convolution and args.convolution[0]:
        net = mx.sym.Reshape(data=net, shape=(args.batch_size, 1, loader.input_dim, 1))
        layer_list = list(range(0, len(args.convolution), 3))
        for l, i in enumerate(layer_list):
            nb_filter = args.convolution[i]
            filter_len = args.convolution[i+1]
            stride = args.convolution[i+2]
            if nb_filter <= 0 or filter_len <= 0 or stride <= 0:
                break
            net = mx.sym.Convolution(data=net, num_filter=nb_filter, kernel=(filter_len, 1), stride=(stride, 1))
            net = mx.sym.Activation(data=net, act_type=args.activation)
            if args.pool:
                net = mx.sym.Pooling(data=net, pool_type="max", kernel=(args.pool, 1), stride=(1, 1))
        net = mx.sym.Flatten(data=net)

    for layer in args.dense:
        if layer:
            net = mx.sym.FullyConnected(data=net, num_hidden=layer)
            net = mx.sym.Activation(data=net, act_type=args.activation)
        if args.drop:
            net = mx.sym.Dropout(data=net, p=args.drop)
    net = mx.sym.FullyConnected(data=net, num_hidden=1)
    net = mx.symbol.LinearRegressionOutput(data=net, label=out)

    plot_network(net, 'net'+ext)

    train_iter = ConcatDataIter(loader, batch_size=args.batch_size, num_data=args.train_samples)
    val_iter = ConcatDataIter(loader, partition='val', batch_size=args.batch_size, num_data=args.val_samples)

    devices = mx.cpu()
    if args.gpus:
        devices = [mx.gpu(i) for i in args.gpus]

    mod = mx.mod.Module(net,
                        data_names=('concat_features',),
                        label_names=('growth',),
                        context=devices)

    initializer = mx.init.Xavier(factor_type="in", magnitude=2.34)
    mod.fit(train_iter, eval_data=val_iter,
            eval_metric=args.loss,
            optimizer=args.optimizer,
            num_epoch=args.epochs,
            initializer=initializer,
            batch_end_callback = mx.callback.Speedometer(args.batch_size, 20))


if __name__ == '__main__':
    main()
