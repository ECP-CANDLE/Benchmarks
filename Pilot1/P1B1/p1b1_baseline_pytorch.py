from __future__ import print_function

import warnings

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from scipy.stats.stats import pearsonr
from sklearn.manifold import TSNE
from torch.autograd import Variable

# import torch.nn.functional as F


with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    from sklearn.metrics import r2_score
    from sklearn.metrics import accuracy_score

import matplotlib as mpl

mpl.use("Agg")
# import candle_pytorch as candle
import candle
import matplotlib.pyplot as plt
import p1b1

np.set_printoptions(precision=4)


def initialize_parameters():

    # Build benchmark object
    p1b1Bmk = p1b1.BenchmarkP1B1(
        p1b1.file_path,
        "p1b1_default_model.txt",
        "pytorch",
        prog="p1b1_baseline",
        desc="Train Autoencoder - Pilot 1 Benchmark 1",
    )

    # Initialize parameters
    gParameters = candle.finalize_parameters(p1b1Bmk)
    # p1b1.logger.info('Params: {}'.format(gParameters))

    return gParameters


def save_cache(
    cache_file, x_train, y_train, x_val, y_val, x_test, y_test, x_labels, y_labels
):
    with h5py.File(cache_file, "w") as hf:
        hf.create_dataset("x_train", data=x_train)
        hf.create_dataset("y_train", data=y_train)
        hf.create_dataset("x_val", data=x_val)
        hf.create_dataset("y_val", data=y_val)
        hf.create_dataset("x_test", data=x_test)
        hf.create_dataset("y_test", data=y_test)
        hf.create_dataset(
            "x_labels",
            (len(x_labels), 1),
            "S100",
            data=[x.encode("ascii", "ignore") for x in x_labels],
        )
        hf.create_dataset(
            "y_labels",
            (len(y_labels), 1),
            "S100",
            data=[x.encode("ascii", "ignore") for x in y_labels],
        )


def load_cache(cache_file):
    with h5py.File(cache_file, "r") as hf:
        x_train = hf["x_train"][:]
        y_train = hf["y_train"][:]
        x_val = hf["x_val"][:]
        y_val = hf["y_val"][:]
        x_test = hf["x_test"][:]
        y_test = hf["y_test"][:]
        x_labels = [x[0].decode("unicode_escape") for x in hf["x_labels"][:]]
        y_labels = [x[0].decode("unicode_escape") for x in hf["y_labels"][:]]
    return x_train, y_train, x_val, y_val, x_test, y_test, x_labels, y_labels


class p1b1Model(nn.Module):
    def __init__(self, params, input_dim, cond_dim, seed):
        super(p1b1Model, self).__init__()

        self.keras_defaults = candle.keras_default_config()
        self.seed = seed
        self.winit_func = params["initialization"]

        activation = candle.build_pytorch_activation(params["activation"])
        dropout = params["dropout"]
        dense_layers = params["dense"]
        #    dropout_layer = keras.layers.noise.AlphaDropout if params['alpha_dropout'] else Dropout
        latent_dim = params["latent_dim"]

        if dense_layers is not None:
            if type(dense_layers) != list:
                dense_layers = list(dense_layers)

        # Define model
        # Add layers
        self.ly = nn.Sequential()
        # Encoder Part
        lprev = input_dim
        for i, l in enumerate(dense_layers):
            self.ly.add_module("en_dense%d" % i, nn.Linear(lprev, l))
            self.ly.add_module("en_act%d" % i, activation)
            if params["batch_normalization"]:
                self.ly.add_module("en_bn%d" % i, nn.BatchNorm1d(l))
            if dropout > 0:
                self.ly.add_module("en_dropout%d", nn.Dropout(dropout))
            lprev = l

        if params["model"] == "ae":
            self.ly.add_module("en_dense_latent", nn.Linear(lprev, latent_dim))
            self.ly.add_module("en_act_latent", activation)
            lprev = latent_dim

        # Decoder Part
        output_dim = input_dim
        for i, l in reversed(list(enumerate(dense_layers))):
            self.ly.add_module("de_dense%d" % i, nn.Linear(lprev, l))
            self.ly.add_module("de_act%d" % i, activation)
            if params["batch_normalization"]:
                self.ly.add_module("de_bn%d" % i, nn.BatchNorm1d(l))
            if dropout > 0:
                self.ly.add_module("de_dropout_%d" % i, nn.Dropout(dropout))
            lprev = l

        self.ly.add_module("out_dense", nn.Linear(lprev, output_dim))
        self.ly.add_module("out_act", activation)
        self.reset_parameters()

    def reset_parameters(self):
        """Resets parameters of all the layers."""
        for ly in self.ly:
            if isinstance(ly, nn.Linear):
                candle.pytorch_initialize(
                    ly.weight, self.winit_func, self.keras_defaults, self.seed
                )
                candle.pytorch_initialize(ly.bias, "constant", self.keras_defaults, 0.0)

    def forward(self, x):
        return self.ly(x)


def fit(model, X_train, X_val, params):
    # Training set
    train_data = torch.from_numpy(X_train)
    train_tensor = data.TensorDataset(train_data, train_data)
    train_iter = data.DataLoader(
        train_tensor, batch_size=params["batch_size"], shuffle=params["shuffle"]
    )

    # Validation set
    val_data = torch.from_numpy(X_val)
    val_tensor = torch.utils.data.TensorDataset(val_data, val_data)
    val_iter = torch.utils.data.DataLoader(
        val_tensor, batch_size=params["batch_size"], shuffle=params["shuffle"]
    )

    # Configure GPUs
    # use_gpu = torch.cuda.is_available()
    device_ids = []
    ndevices = torch.cuda.device_count()
    if ndevices > 1:
        for i in range(ndevices):
            device_i = torch.device("cuda:" + str(i))
            device_ids.append(device_i)
        device = device_ids[0]
    elif ndevices == 1:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    # Instantiate with parallel processing
    if ndevices > 1:
        model = nn.DataParallel(model, device_ids, device)
    model.to(device)

    loss_fn = candle.get_pytorch_function(params["loss"])

    # Train the model
    freq_log = 1

    if params["learning_rate"] is None:
        learning_rate = 1e-2
    optimizer = candle.build_pytorch_optimizer(
        model, params["optimizer"], learning_rate, model.keras_defaults
    )
    ckpt = candle.CandleCkptPyTorch(params)
    ckpt.set_model({"model": model, "optimizer": optimizer})

    total_step = len(train_iter)
    loss_list = []
    acc_list = []
    for epoch in range(params["epochs"]):
        train_loss = 0
        for batch, (in_train, _) in enumerate(train_iter):
            # in_train = Variable(in_train)
            # if use_gpu:
            #    in_train = in_train.cuda()
            if ndevices > 0:
                in_train = in_train.to(device)

            # Run the forward pass
            output = model(in_train)
            loss = loss_fn(output, in_train)
            loss_list.append(loss.item())

            # Backprop and perform optimisation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()  # loss.data[0]

            # Logging
            if batch % freq_log == 0:
                print(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                        epoch,
                        batch * len(in_train),
                        len(train_iter.dataset),
                        100.0 * batch / len(train_iter),
                        loss.item(),
                    )
                )
                # loss.data[0]))# / len(in_train)))
            # end batch loop
        # epoch loop
        print(
            "====> Epoch: {} Average loss: {:.4f}".format(
                epoch, train_loss / len(train_iter.dataset)
            )
        )
        ckpt.ckpt_epoch(epoch, train_loss)
        # end epoch loop


def run(params):

    args = candle.ArgumentStruct(**params)
    seed = args.rng_seed
    candle.set_seed(seed)

    # Construct extension to save model
    ext = p1b1.extension_from_parameters(params, ".pytorch")
    candle.verify_path(params["save_path"])
    prefix = "{}{}".format(params["save_path"], ext)
    logfile = params["logfile"] if params["logfile"] else prefix + ".log"
    candle.set_up_logger(logfile, p1b1.logger, params["verbose"])
    # p1b1.logger.info('Params: {}'.format(params))

    # Get default parameters for initialization and optimizer functions
    keras_defaults = candle.keras_default_config()

    # Load dataset
    x_train, y_train, x_val, y_val, x_test, y_test, x_labels, y_labels = p1b1.load_data(
        params, seed
    )

    # cache_file = 'data_l1000_cache.h5'
    # save_cache(cache_file, x_train, y_train, x_val, y_val, x_test, y_test, x_labels, y_labels)
    # x_train, y_train, x_val, y_val, x_test, y_test, x_labels, y_labels = load_cache(cache_file)

    p1b1.logger.info("Shape x_train: {}".format(x_train.shape))
    p1b1.logger.info("Shape x_val:   {}".format(x_val.shape))
    p1b1.logger.info("Shape x_test:  {}".format(x_test.shape))

    p1b1.logger.info(
        "Range x_train: [{:.3g}, {:.3g}]".format(np.min(x_train), np.max(x_train))
    )
    p1b1.logger.info(
        "Range x_val:   [{:.3g}, {:.3g}]".format(np.min(x_val), np.max(x_val))
    )
    p1b1.logger.info(
        "Range x_test:  [{:.3g}, {:.3g}]".format(np.min(x_test), np.max(x_test))
    )

    p1b1.logger.debug("Class labels")
    for i, label in enumerate(y_labels):
        p1b1.logger.debug("  {}: {}".format(i, label))

    # clf = build_type_classifier(x_train, y_train, x_val, y_val)

    n_classes = len(y_labels)
    cond_train = y_train
    cond_val = y_val
    cond_test = y_test

    input_dim = x_train.shape[1]
    cond_dim = cond_train.shape[1]

    model = p1b1Model(params, input_dim, cond_dim, seed)
    # Display model
    print(model)
    # Train model
    fit(model, x_train, x_val, params)


def main():
    params = initialize_parameters()
    run(params)


if __name__ == "__main__":
    main()
