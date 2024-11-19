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

from timeit import default_timer as timer

class myNeuralNetwork(nn.Module):

    def __init__(self):
        super().__init__()

        self.act = nn.ReLU()

        self.conv = nn.Conv1d(1, 128, 20)
        self.pool1 = nn.MaxPool1d(1)

        self.conv2 = nn.Conv1d(128, 128, 10)
        self.pool2 = nn.MaxPool1d(10)

        self.drop = nn.Dropout(0.1)

        self.lin = nn.Linear(773760, 200)
        self.lin2 = nn.Linear(200, 20)
        self.lin3 = nn.Linear(20, 36)

    def forward(self, x):
        x = self.conv(x)
        x = self.act(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.act(x)
        x = self.pool2(x)
        
        x = torch.flatten(x, start_dim=1)
        
        x = self.lin(x)
        x = self.act(x)
        x = self.drop(x)
        
        x = self.lin2(x)
        x = self.act(x)
        x = self.drop(x)
        
        x = self.lin3(x)

        return x

gParameters = initialize_parameters()
X_train, Y_train, X_test, Y_test = bmk.load_data(gParameters)

x_train_len = X_train.shape[1]

X_train = np.expand_dims(X_train, axis=1)
X_test = np.expand_dims(X_test, axis=1)

output_dir = gParameters['output_dir']
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

model = myNeuralNetwork()
print(model)

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True)
criterion = nn.CrossEntropyLoss()
epochs = gParameters['epochs']
#batchsize = gParameters['batch_size']
batchsize = 16

train_data = []
for i in range(len(X_train)):
    train_data.append([X_train[i], Y_train[i]])

trainloader = DataLoader(train_data, batch_size=batchsize)

test_data = []
for i in range(len(X_test)):
    test_data.append([X_test[i], Y_test[i]])

testloader = DataLoader(test_data, batch_size=batchsize)

start = timer()

for epoch in range(epochs):
    model.train()
    #running_loss = 0.0
    correct = 0
    for i, (inputs, targets) in enumerate(trainloader):
        #print('t:', targets.shape)
        optimizer.zero_grad()
        output = model(inputs)
        _, predicted = torch.max(output.data, 1)
        _, label = torch.max(targets.data, 1)
        #print('o:', output.shape)
        loss = criterion(output, torch.max(targets, 1)[1])
        #running_loss += loss.item()
        loss.backward()
        optimizer.step()
        #scheduler.step(running_loss/len(trainloader))
        correct += (predicted == label).sum().item()
        #print('Finished Epoch %d Batch %d' % (epoch, i))
    
    #output = (output>0.5).float()
    #correct = (output == targets).float().sum().item()
    #print(loss, correct, len(train_data), 'should be 4000 ish')
    with torch.no_grad():
        val_correct = 0
        val_total = 0
        model.eval()
        for i, (inputs, targets) in enumerate(testloader):
            output = model(inputs)
            _, predicted = torch.max(output.data, 1)
            #print(predicted.shape, targets.shape)#20, 20,36
            _, label = torch.max(targets.data, 1)
            val_loss = criterion(output, torch.max(targets, 1)[1])
            #print(label.shape)
            #print('predicted:', predicted)
            #print('targets:', label)
            val_total += label.size(0)
            val_correct += (predicted == label).sum().item()
            #print('Finished Batch %d' % i)
    print('Epoch {}/{}, Loss: {:.3f}, Accuracy: {:.3f}, Val Loss: {:.3f}, Val Accuracy: {:.3f}'.format(epoch+1, epochs, loss.item(), correct/len(train_data)*100, val_loss.item(), val_correct/val_total*100))
    scheduler.step(val_loss)
    
print('training done')

with torch.no_grad():
    correct = 0
    total = 0
    model.eval()
    for i, (inputs, targets) in enumerate(testloader):
        output = model(inputs)
        _, predicted = torch.max(output.data, 1)
        #print(predicted.shape, targets.shape)#20, 20,36
        _, label = torch.max(targets.data, 1)
        #print(label.shape)
        #print('predicted:', predicted)
        #print('targets:', label)
        total += label.size(0)
        correct += (predicted == label).sum().item()
        #print('Finished Batch %d' % i)

end = timer()

print('Final Accuracy: %.2f %%' % (100 * correct / total))
print('Final Time: %d seconds' % (end - start))
