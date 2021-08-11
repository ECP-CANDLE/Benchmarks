#Alex Nicolellis
from torch.optim import lr_scheduler
from tc1_baseline_keras2 import initialize_parameters
import numpy as np
import os
import sys

import torch
from torch import nn , optim
from torch.utils.data import DataLoader

file_path = os.path.dirname(os.path.realpath(__file__))
lib_path2 = os.path.abspath(os.path.join(file_path, '..', '..', 'common'))

sys.path.append(lib_path2)
import tc1 as bmk
import candle
import pytorch_utils as utils

from timeit import default_timer as timer

gParameters = initialize_parameters()

class myNeuralNetwork(nn.Module):

    def __init__(self):
        super().__init__()

        self.dense_first = True
        self.act = utils.build_activation(gParameters['activation'])
        if gParameters['dropout']:
            self.dropout = nn.Dropout(gParameters['dropout'])
        layer_list = list(range(0, len(gParameters['conv']), 3))
        self.convs = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.linears = nn.ModuleList()
        for _, i in enumerate(layer_list):
            out_channels = gParameters['conv'][i]
            kernel_size = gParameters['conv'][i + 1]
            stride = gParameters['conv'][i + 2]
            #print(i / 3, out_channels, kernel_size, stride)
            if gParameters['pool']:
                pool_list = gParameters['pool']
                if type(pool_list) != list:
                    pool_list = list(pool_list)
            if out_channels <= 0 or kernel_size <= 0 or stride <= 0:
                break
            self.dense_first = False
            if 'locally_connected' in gParameters:
                #model.add(LocallyConnected1D(filters, filter_len, strides=stride,
                                            #padding='valid', input_shape=(x_train_len, 1)))
                pass
            else:
                # input layer
                if i == 0:
                    #print('First conv', 1, out_channels, kernel_size, stride)
                    self.convs.append(nn.Conv1d(1, out_channels=out_channels, kernel_size=kernel_size,
                                    stride=stride))
                else:
                    in_channels = gParameters['conv'][i-3]
                    #print('Another conv', in_channels, out_channels, kernel_size, stride)
                    self.convs.append(nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                    stride=stride))
            #print(self.act)
            if gParameters['pool']:
                #print('Pool', pool_list[i//3])
                self.pools.append(nn.MaxPool1d(kernel_size=pool_list[i // 3]))
        
        self.flat = nn.Flatten(start_dim=1)

    def forward(self, x):
        if not self.dense_first:
            for i in range(len(self.convs)):
                x = self.convs[i](x)
                x = self.act(x)
                if gParameters['pool']:
                    x = self.pools[i](x)
            x = self.flat(x)
        
        #should only run once: create linear layers
        #this is in the forward method because we need access to x.size() to define the first linear layer
        if len(self.linears) == 0:
            for i, layer in enumerate(gParameters['dense']):
                if layer:
                    if i == 0:
                        input_shape = x.size()[1]
                        #print('Linear', input_shape, layer)
                        self.linears.append(nn.Linear(in_features=input_shape, out_features=layer))
                    else:
                        #print('Linear', gParameters['dense'][i-1], layer)
                        self.linears.append(nn.Linear(in_features=gParameters['dense'][i-1], out_features=layer))
                    #print('Act')
            self.linears.append(nn.Linear(in_features=gParameters['dense'][-1], out_features=gParameters['classes']))
        
        for i in range(len(self.linears)-1):
            x = self.linears[i](x)
            x = self.act(x)
            if gParameters['dropout']:
                x = self.dropout(x)

        if self.dense_first:
            x = self.flat(x)
        
        x = self.linears[-1](x)

        return x

#preprocess data
X_train, Y_train, X_test, Y_test = bmk.load_data(gParameters)

x_train_len = X_train.shape[1]

X_train = np.expand_dims(X_train, axis=1)
X_test = np.expand_dims(X_test, axis=1)

output_dir = gParameters['output_dir']
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

model = myNeuralNetwork()
print(model)

kerasDefaults = {'momentum_sgd': 0.0, 'nesterov_sgd': False, 'rho' : 0.9, 'epsilon' : 1e-07, 
                'beta_1': 0.9, 'beta_2': 0.999}

# the utils function is outdated
if gParameters['optimizer'] == 'sgd':
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
else:
    optimizer = utils.build_optimizer(model, gParameters['optimizer'], 0.01, kerasDefaults)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True)
criterion = nn.CrossEntropyLoss()
epochs = gParameters['epochs']
batchsize = gParameters['batch_size']

#put data into Dataloaders for easier parsing later
train_data = []
for i in range(len(X_train)):
    train_data.append([X_train[i], Y_train[i]])

trainloader = DataLoader(train_data, batch_size=batchsize)

test_data = []
for i in range(len(X_test)):
    test_data.append([X_test[i], Y_test[i]])

testloader = DataLoader(test_data, batch_size=batchsize)

start = timer()

#training
for epoch in range(epochs):
    model.train()
    val_correct = 0
    val_total = 0
    correct = 0
    for i, (inputs, targets) in enumerate(trainloader):
        optimizer.zero_grad()
        output = model(inputs)
        _, predicted = torch.max(output.data, 1)
        _, label = torch.max(targets.data, 1)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        correct += (predicted == label).sum().item()
    
    #get validation data
    with torch.no_grad():
        model.eval()
        for i, (inputs, targets) in enumerate(testloader):
            output = model(inputs)
            _, predicted = torch.max(output.data, 1)
            _, label = torch.max(targets.data, 1)
            val_loss = criterion(output, label)
            val_total += label.size(0)
            val_correct += (predicted == label).sum().item()

    print('Epoch {}/{}, Loss: {:.3f}, Accuracy: {:.3f}, Val Loss: {:.3f}, Val Accuracy: {:.3f}'.format(epoch+1, 
            epochs, loss.item(), correct/len(train_data)*100, val_loss.item(), val_correct/val_total*100))
    scheduler.step(val_loss)
    
print('training done')

#testing
with torch.no_grad():
    correct = 0
    total = 0
    model.eval()
    for i, (inputs, targets) in enumerate(testloader):
        output = model(inputs)
        _, predicted = torch.max(output.data, 1)
        _, label = torch.max(targets.data, 1)
        total += label.size(0)
        correct += (predicted == label).sum().item()

end = timer()

print('Final Accuracy: %.2f %%' % (100 * correct / total))
print('Final Time: %d seconds' % (end - start))