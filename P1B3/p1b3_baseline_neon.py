from __future__ import division, print_function

import argparse
import logging

import numpy as np

import neon
from neon.util.argparser import NeonArgparser
from neon.callbacks.callbacks import Callbacks
from neon.initializers import Gaussian, GlorotUniform
from neon.layers import GeneralizedCost, Affine, Conv, Dropout, Pooling, Reshape
from neon.models import Model
from neon.optimizers import GradientDescentMomentum
from neon.transforms import Identity
# from neon.transforms import MeanSquared

from neon import transforms


import p1b3


# Model and Training parameters

# Seed for random generation
SEED = 2017
# Size of batch for training
BATCH_SIZE = 100
# Number of training epochs
NB_EPOCH = 20
# Number of data generator workers
NB_WORKER = 1

# Percentage of dropout used in training
DROP = 0.1
# Activation function (options: 'relu', 'tanh', 'sigmoid', 'linear')
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
    params = {'batch_size': BATCH_SIZE, 'epochs': NB_EPOCH}

    parser = NeonArgparser(__doc__, default_overrides=params)

    # parser = argparse.ArgumentParser(prog='p1b3_baseline',
    #                                  formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument("-v", "--verbose", action="store_true",
                        # help="increase output verbosity")
    parser.add_argument("-a", "--activation", action="store",
                        default=ACTIVATION,
                        help="keras activation function to use in inner layers: relu, tanh, sigmoid...")
    # parser.add_argument("-b", "--batch_size", action="store",
    #                     default=BATCH_SIZE, type=int,
    #                     help="batch size")
    parser.add_argument("--convolution", action="store", nargs='+', type=int,
                        default=CONVOLUTION_LAYERS,
                        help="integer array describing convolution layers: conv1_nb_filter, conv1_filter_len, conv1_stride, conv2_nb_filter, conv2_filter_len, conv2_stride ...")
    parser.add_argument("--dense", action="store", nargs='+', type=int,
                        default=DENSE_LAYERS,
                        help="number of units in fully connected layers in an integer array")
    # parser.add_argument("-e", "--epochs", action="store",
    #                     default=NB_EPOCH, type=int,
    #                     help="number of training epochs")
    parser.add_argument("--locally_connected", action="store_true",
                        default=False,  # TODO: not currently supported
                        help="use locally connected layers instead of convolution layers")
    parser.add_argument("--optimizer", action="store",
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
    ext = '.neon'
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


class ConcatDataIter(neon.NervanaObject):
    """
    Data iterator for concatenated features
    Modeled after ArrayIterator: https://github.com/NervanaSystems/neon/blob/master/neon/data/dataiterator.py
    """

    def __init__(self, data_loader,
                 partition='train',
                 ndata=None,
                 lshape=None,
                 datatype=np.float32):
        """
        During initialization, the input data will be converted to backend tensor objects
        (e.g. CPUTensor or GPUTensor). If the backend uses the GPU, the data is copied over to the
        device.
        """
        super(ConcatDataIter, self).__init__()
        self.data = data_loader
        self.gen = p1b3.DataGenerator(data_loader, partition=partition, batch_size=self.be.bsz, concat=True)
        self.ndata = ndata or self.gen.num_data
        assert self.ndata >= self.be.bsz
        self.datatype = datatype
        self.gen = self.gen.flow()
        self.start = 0
        self.ybuf = None
        self.shape = lshape or data_loader.input_dim
        self.lshape = lshape

    @property
    def nbatches(self):
        """
        Return the number of minibatches in this dataset.
        """
        return (self.ndata - self.start) // self.be.bsz

    def reset(self):
        self.start = 0

    def __iter__(self):
        """
        Returns a new minibatch of data with each call.

        Yields:
            tuple: The next minibatch which includes both features and labels.
        """

        def transpose_gen(z):
            return (self.be.array(z), self.be.iobuf(z.shape[1]),
                    lambda _in, _out: self.be.copy_transpose(_in, _out))

        for i1 in range(self.start, self.ndata, self.be.bsz):
            bsz = min(self.be.bsz, self.ndata - i1)
            # islice1, oslice1 = slice(0, bsz), slice(i1, i1 + bsz)
            islice1, oslice1 = slice(0, bsz), slice(0, bsz)
            islice2, oslice2 = None, None
            if self.be.bsz > bsz:
                islice2, oslice2 = slice(bsz, None), slice(0, self.be.bsz - bsz)
                self.start = self.be.bsz - bsz

            x, y = next(self.gen)
            x = np.ascontiguousarray(x).astype(self.datatype)
            y = np.ascontiguousarray(y).astype(self.datatype)

            X = [x]
            y = y.reshape(y.shape + (1,))

            self.Xdev, self.Xbuf, self.unpack_func = list(zip(*[transpose_gen(x) for x in X]))
            self.dbuf, self.hbuf = list(self.Xdev), list(self.Xbuf)
            self.unpack_func = list(self.unpack_func)

            self.ydev, self.ybuf, yfunc = transpose_gen(y)
            self.dbuf.append(self.ydev)
            self.hbuf.append(self.ybuf)
            self.unpack_func.append(yfunc)

            for buf, dev, unpack_func in zip(self.hbuf, self.dbuf, self.unpack_func):
                unpack_func(dev[oslice1], buf[:, islice1])
                if oslice2:
                    unpack_func(dev[oslice2], buf[:, islice2])

            inputs = self.Xbuf[0] if len(self.Xbuf) == 1 else self.Xbuf
            targets = self.ybuf if self.ybuf else inputs

            yield (inputs, targets)


def get_function(name):
    mapping = {}

    # activation
    mapping['relu'] = neon.transforms.activation.Rectlin
    mapping['sigmoid'] = neon.transforms.activation.Logistic
    mapping['tanh'] = neon.transforms.activation.Tanh
    mapping['linear'] = neon.transforms.activation.Identity

    # loss
    mapping['mse'] = neon.transforms.cost.MeanSquared
    mapping['binary_crossentropy'] = neon.transforms.cost.CrossEntropyBinary
    mapping['categorical_crossentropy'] = neon.transforms.cost.CrossEntropyMulti

    # optimizer
    def SGD(learning_rate=0.01, momentum_coef=0.9, gradient_clip_value=5):
        return GradientDescentMomentum(learning_rate, momentum_coef, gradient_clip_value)

    mapping['sgd'] = SGD
    mapping['rmsprop'] = neon.optimizers.optimizer.RMSProp
    mapping['adam'] = neon.optimizers.optimizer.Adam
    mapping['adagrad'] = neon.optimizers.optimizer.Adagrad
    mapping['adadelta'] = neon.optimizers.optimizer.Adadelta

    mapped = mapping.get(name)
    if not mapped:
        raise Exception('No neon function found for "{}"'.format(name))

    return mapped


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

    # initializer = Gaussian(loc=0.0, scale=0.01)
    initializer = GlorotUniform()
    activation = get_function(args.activation)()

    layers = []
    reshape = None

    if args.convolution and args.convolution[0]:
        reshape = (1, loader.input_dim, 1)
        layer_list = list(range(0, len(args.convolution), 3))
        for l, i in enumerate(layer_list):
            nb_filter = args.convolution[i]
            filter_len = args.convolution[i+1]
            stride = args.convolution[i+2]
            # print(nb_filter, filter_len, stride)
            # fshape: (height, width, num_filters).
            layers.append(Conv((1, filter_len, nb_filter), strides={'str_h':1, 'str_w':stride}, init=initializer, activation=activation))
            if args.pool:
                layers.append(Pooling((1, args.pool)))

    for layer in args.dense:
        if layer:
            layers.append(Affine(nout=layer, init=initializer, activation=activation))
        if args.drop:
            layers.append(Dropout(keep=(1-args.drop)))
    layers.append(Affine(nout=1, init=initializer, activation=neon.transforms.Identity()))

    model = Model(layers=layers)

    train_iter = ConcatDataIter(loader, ndata=args.train_samples, lshape=reshape, datatype=args.datatype)
    val_iter = ConcatDataIter(loader, partition='val', ndata=args.val_samples, lshape=reshape, datatype=args.datatype)

    cost = GeneralizedCost(get_function(args.loss)())
    optimizer = get_function(args.optimizer)()
    callbacks = Callbacks(model, eval_set=val_iter, **args.callback_args)

    model.fit(train_iter, optimizer=optimizer, num_epochs=args.epochs, cost=cost, callbacks=callbacks)


if __name__ == '__main__':
    main()
