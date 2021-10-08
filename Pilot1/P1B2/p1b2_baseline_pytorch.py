from __future__ import print_function

import numpy as np

import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable

import p1b2
import candle


def initialize_parameters(default_model='p1b2_default_model.txt'):

    # Build benchmark object
    p1b2Bmk = p1b2.BenchmarkP1B2(p1b2.file_path, default_model, 'pytorch',
                                 prog='p1b2_baseline', desc='Train Classifier - Pilot 1 Benchmark 2')

    # Initialize parameters
    gParameters = candle.finalize_parameters(p1b2Bmk)

    return gParameters


class p1b2Model(nn.Module):
    def __init__(self, params, input_dim, output_dim, seed):
        super(p1b2Model, self).__init__()

        self.keras_defaults = candle.keras_default_config()
        self.seed = seed
        self.winit_func = params['initialization']

        activation = candle.build_activation(params['activation'])
        dropout = params['dropout']

        # Define MLP architecture
        ly = []
        layers = params['dense']
        lprev = input_dim

        if layers is not None:
            if type(layers) != list:
                layers = list(layers)
            for i, l in enumerate(layers):
                ly.append(nn.Linear(lprev, l))
                ly.append(activation)
                lprev = l
                if dropout > 0.:
                    ly.append(nn.Dropout(dropout))
            ly.append(nn.Linear(lprev, output_dim))
        else:
            ly.append(nn.Linear(lprev, output_dim))

        self.ly = nn.ModuleList(ly)
        #self.reset_parameters()


    def reset_parameters(self):
        """ Resets parameters of all the layers. """
        for ly in self.ly:
            if isinstance(ly, nn.Linear):
                candle.initialize(ly.weight, self.winit_func, self.keras_defaults, self.seed)
                candle.initialize(ly.bias, 'constant', self.keras_defaults, 0.0)


    def forward(self, x):
        for ly in self.ly:
            x = ly(x)
        return x


class Loss_regl2(torch.nn.Module):

    def __init__(self, loss_aux, reg_l2, model, ndevices):
        super(Loss_regl2,self).__init__()
        self.reg_l2 = reg_l2
        self.model_ = model
        self.ndevices = ndevices
        self.loss = loss_aux

    def forward(self, x, y):

        if self.ndevices > 0:
            regularization_loss = Variable(torch.zeros(1)).float().cuda()
        else:
            regularization_loss = Variable(torch.zeros(1)).float()
        for param in self.model_.parameters():
            if isinstance(param, nn.Linear):
                regularization_loss += torch.sum(torch.norm(param.weight.data))

        if self.ndevices > 0:
            loss_ = self.loss.cuda()
        else:
            loss_ = self.loss
        totloss = loss_(x,y) + regularization_loss * self.reg_l2
        return totloss



def fit(model, X_train, X_val, y_train, y_val, params):

    # Training set
    train_xdata = torch.from_numpy(X_train)
    train_ydata = torch.from_numpy(y_train)
    train_tensor = data.TensorDataset(train_xdata, train_ydata)
    train_iter = data.DataLoader(train_tensor, batch_size=params['batch_size'], shuffle=params['shuffle'])

    # Validation set
    val_xdata = torch.from_numpy(X_val)
    val_ydata = torch.from_numpy(y_val)
    val_tensor = torch.utils.data.TensorDataset(val_xdata, val_ydata)
    val_iter = torch.utils.data.DataLoader(val_tensor, batch_size=params['batch_size'], shuffle=False)

    # Configure GPUs
    device_ids = []
    ndevices = torch.cuda.device_count()
    if ndevices > 1:
        for i in range(ndevices):
            device_i = torch.device('cuda:'+str(i))
            device_ids.append(device_i)
        device = device_ids[0]
    elif ndevices == 1:
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    # Instantiate with parallel processing
    if ndevices > 1:
        model = nn.DataParallel(model, device_ids, device)
    model.to(device)


    # Define optimizer
    optimizer = candle.build_optimizer(model, params['optimizer'],
                                       params['learning_rate'],
                                       model.keras_defaults)

    # Define loss function
    if params['reg_l2'] > 0.:
        loss_aux = candle.get_function(params['loss'])
        loss_fn = Loss_regl2(loss_aux, params['reg_l2'], model, ndevices)
    else:
        loss_fn = candle.get_function(params['loss'])

    # Train the model
    freq_log = 10

    total_step = len(train_iter)
    loss_list = []
    acc_list = []
    for epoch in range(params['epochs']):
        train_loss = 0
        for batch, (xtrain, ytrain) in enumerate(train_iter):
            if ndevices > 0:
                xtrain = xtrain.to(device)

            # Run the forward pass
            output = model(xtrain)
            loss = loss_fn(output, ytrain)
            loss_list.append(loss.item())

            # Backprop and perform optimisation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()  # loss.data[0]

            # Logging
            if batch % freq_log == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch * len(xtrain), len(train_iter.dataset), 100. * batch / len(train_iter), loss.item()))
                        # loss.data[0]))# / len(in_train)))

        print('====> Epoch: {} Average loss: {:.4f}'.format(
            epoch, train_loss / len(train_iter.dataset)))

        # Compute validation Loss
        model.train(False)
        val_loss = 0.0
        for btch, (xval, yval) in enumerate(val_iter):
            if ndevices > 0:
                xval = xval.to(device)
            # forward pass to get outputs
            output = model(xval)
            # calculate the loss between predicted and target keypoints
            loss = loss_fn(output, yval)
            val_loss += loss.item()
        print('=> Average validation loss: {:.4f}'.format(val_loss / len(val_iter.dataset)))
        model.train(True)

    if ndevices > 0:
        return model.cpu()

    return model


def run(gParameters):

    # Construct extension to save model
    ext = p1b2.extension_from_parameters(gParameters, '.torch')
    candle.verify_path(gParameters['save_path'])
    prefix = '{}{}'.format(gParameters['save_path'], ext)
    logfile = gParameters['logfile'] if gParameters['logfile'] else prefix + '.log'
    candle.set_up_logger(logfile, p1b2.logger, gParameters['verbose'])
    p1b2.logger.info('Params: {}'.format(gParameters))

    # Get default parameters for initialization and optimizer functions
    kerasDefaults = candle.keras_default_config()
    seed = gParameters['rng_seed']
    candle.set_seed(seed)

    # Load dataset
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = p1b2.load_data_one_hot(gParameters, seed)
    # pytroch needs classes
    y_train_pt = candle.convert_to_class(y_train)
    y_val_pt = candle.convert_to_class(y_val)
    #y_test_pt = candle.convert_to_class(y_test)

    p1b2.logger.info("Shape X_train: {}".format(X_train.shape))
    p1b2.logger.info("Shape X_val:   {}".format(X_val.shape))
    p1b2.logger.info("Shape X_test:  {}".format(X_test.shape))
    p1b2.logger.info("Shape y_train: {}".format(y_train.shape))
    p1b2.logger.info("Shape y_val:   {}".format(y_val.shape))
    p1b2.logger.info("Shape y_test:  {}".format(y_test.shape))

    p1b2.logger.info("Range X_train: [{:.3g}, {:.3g}]".format(np.min(X_train), np.max(X_train)))
    p1b2.logger.info("Range X_val:   [{:.3g}, {:.3g}]".format(np.min(X_val), np.max(X_val)))
    p1b2.logger.info("Range X_test:  [{:.3g}, {:.3g}]".format(np.min(X_test), np.max(X_test)))
    p1b2.logger.info("Range y_train: [{:.3g}, {:.3g}]".format(np.min(y_train), np.max(y_train)))
    p1b2.logger.info("Range y_val:   [{:.3g}, {:.3g}]".format(np.min(y_val), np.max(y_val)))
    p1b2.logger.info("Range y_test:  [{:.3g}, {:.3g}]".format(np.min(y_test), np.max(y_test)))

    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1]

    # Construct model
    mlp = p1b2Model(gParameters, input_dim, output_dim, seed)

    # Display model
    print(mlp)

    # Train model
    mlp = fit(mlp, X_train, X_val, y_train_pt, y_val_pt, gParameters)

    # model save
    # save_filepath = "model_mlp_" + ext
    # torch.save({'state_dict': mlp.state_dict()}, save_filepath)

    # Evalute model on test set
    mlp.train(False)
    y_pred = mlp(torch.from_numpy(X_test)).detach().numpy()
    scores = p1b2.evaluate_accuracy_one_hot(y_pred, y_test)
    print('Evaluation on test data:', scores)


def main():
    params = initialize_parameters()
    run(params)


if __name__ == '__main__':
    main()
