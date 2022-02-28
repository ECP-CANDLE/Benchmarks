import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import time
import sklearn
import sklearn.metrics

from typing import Dict
from dataclasses import dataclass
import argparse

import cnn
from cnn import generate_torch_data

##########################################################################################
# Arguments
##########################################################################################

parser = argparse.ArgumentParser()

parser.add_argument('input_prefix', help='location for data')
parser.add_argument('--model_output_file_prefix', type=str, default=None, help='output model')
parser.add_argument('--saved_model', type=str, default=None, help='saved model')
parser.add_argument('--epochs', type=int, default=1, help='number of epochs to train')
parser.add_argument('--no_train', action='store_true', default=False, help='option to skip training')
parser.add_argument('--validate', action='store_true', default=False, help='option to run validation after each epoch')
parser.add_argument('--test', action='store_true', default=False, help='option to test at end of training')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--new_optimizer', action='store_true', default=False, help='option to construct new optimizer')
parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
parser.add_argument('--device', type=str, default='cuda:0', help='device for training')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay')
parser.add_argument('--test_while_training', action='store_true', default=False, help='option to test while training to generate learning curve')
parser.add_argument('--use_position_weights', action='store_true', default=False, help='option to use position weights')
parser.add_argument('--train_postfix', type=str, default='', help='postfix for training data')

args = parser.parse_args()

print(args)

##########################################################################################
# Load Data
##########################################################################################

def read_tokenized_data(split_name, numberOfClasses):

    postfix = ''
    if split_name == 'train':
        postfix += args.train_postfix

    filename_x = args.input_prefix + '_' + split_name + '_x' + postfix + '.txt'
    filename_y = args.input_prefix + '_' + split_name + '_y' + postfix + '.txt'
    print('Loading %s' % filename_x)
    output_x = []
    with open(filename_x, 'r') as f:
        for row in f:
            output_x.append(np.array(row.split(), dtype=np.int32))

    print('Loading %s' % filename_y)
    output_y = np.zeros((len(output_x), numberOfClasses))
    with open(filename_y, 'r') as f:
        row_counter = 0
        for row in f:
            for index in row.split():
                if int(index) >= 0:
                    output_y[row_counter, int(index)] = 1.0
            row_counter += 1

    print('Total Samples: %d' % len(output_x), flush=True)

    return output_x, output_y

numberOfTokens = 1
with open(args.input_prefix + '_vocab.txt', 'r') as f:
    for row in f:
        numberOfTokens += 1
print('Number of Tokens: ' + str(numberOfTokens), flush=True)

numberOfClasses = 0
class_counts = []
with open(args.input_prefix + '_labels.txt','r') as f:
    for row in f:
        class_counts.append(int(row.split()[-1]))
        numberOfClasses += 1
print('Number of Classes: ' + str(numberOfClasses), flush=True)

# position weights for classes - classes with more positive examples have higher weight
p_weights = torch.ones(numberOfClasses)
if args.use_position_weights:
    p_weights += torch.tensor(class_counts)
    p_weights = torch.log(p_weights)

if not args.no_train:
    train_x, train_y = read_tokenized_data('train', numberOfClasses)

if args.validate:
    val_x, val_y = read_tokenized_data('val', numberOfClasses)

if args.test or args.test_while_training:
    test_x, test_y = read_tokenized_data('test', numberOfClasses)

##########################################################################################
# Model Setup
##########################################################################################

device = torch.device(args.device)
model = cnn.Model(numberOfClasses, cnn.Hparams(vocab_size=numberOfTokens))

pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('Number of Trainable Parameters: ' + str(pytorch_total_params), flush=True)

if args.saved_model is not None:
    checkpoint = torch.load(args.saved_model, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])

model = model.to(device)
p_weights = p_weights.to(device)

optimizer = optim.Adam( model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
if (args.saved_model is not None) and (not args.new_optimizer):
    checkpoint = torch.load(args.saved_model, map_location='cpu')
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

loss_fcn = torch.nn.BCEWithLogitsLoss(pos_weight=p_weights)

if not args.no_train:
    labels = torch.tensor(train_y, dtype=torch.float, device=device)

if args.validate:
    labelsVal = torch.tensor(val_y, dtype=torch.float, device=device)

if args.test or args.test_while_training:
    labelsTest = torch.tensor(test_y, dtype=torch.float, device=device)

batch_size = args.batch_size

##########################################################################################
# Function Definitions
##########################################################################################

def correctPreds(output, labels):
    preds = 1.0*(torch.sigmoid(output).flatten() >= 0.5)
    correct = preds.eq(labels.flatten()).double()
    correct = correct.sum()
    return correct.item()

def train_epoch(model, optimizer, data_x, data_y, batch_size, device):
    model.train()

    t = time.time()
    indicesForTraining = np.arange(len(data_x))
    permutation = torch.randperm(len(indicesForTraining))
    correctSum = 0.0
    totalSum = 0.0
    totalLoss = 0.0

    batch_counter = 0
    for k in range(0, len(permutation), batch_size):
        start_index = k
        end_index = min(len(permutation), k+batch_size)
        indices = indicesForTraining[permutation[start_index:end_index]]
        if end_index == (start_index + 1):
            indices = [indices]
        trainBatch = generate_torch_data(data_x, indices)
        trainBatch = trainBatch.to(device)
        output = model(trainBatch)
        loss_train = loss_fcn(output, data_y[indices])
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()

        correctSum += correctPreds(output.detach(), labels[indices])
        totalSum += len(indices)*numberOfClasses
        totalLoss += loss_train.item()
        batch_counter += 1

    train_time = time.time() - t
    accuracy = correctSum / totalSum
    loss = totalLoss / batch_counter

    return train_time, accuracy, loss

def evaluate_model(model, data_x, data_y, batch_size, device):
    model.eval()

    allClassPreds = np.zeros((len(data_x)*numberOfClasses, 1))
    for k in range(0, len(data_x), batch_size):
        start_index = k
        end_index = min(len(data_x), k+batch_size)
        indices = np.arange(start_index, end_index)
        dataBatch = generate_torch_data(data_x, indices)
        dataBatch = dataBatch.to(device)
        output = torch.sigmoid(model(dataBatch))
        allClassPreds[start_index*numberOfClasses:end_index*numberOfClasses,0] = 1.0*(output.detach().cpu().flatten() >= 0.5)

    precision = sklearn.metrics.precision_score(data_y.detach().cpu().flatten().numpy(), allClassPreds)
    recall = sklearn.metrics.recall_score(data_y.detach().cpu().flatten().numpy(), allClassPreds)
    micro = sklearn.metrics.f1_score(data_y.detach().cpu().flatten().numpy(), allClassPreds)

    return precision, recall, micro

##########################################################################################
# Train the Model
##########################################################################################

maxVal = 0.0

if not args.no_train:
    model.train()
    for i in range(args.epochs):

        print('Epoch: ' + str(i), flush=True)

        t, acc, loss = train_epoch(model, optimizer, train_x, labels, args.batch_size, device)

        print('Train Acc: ', acc,flush=True)
        print('Train Time: ', t, flush=True)
        print('Train Loss: ', loss, flush=True)

        if args.validate:
            p, r, f1 = evaluate_model(model, val_x, labelsVal, args.batch_size, device)
            print('Val Precision: %0.6f' % p)
            print('Val Recall: %0.6f' % r)
            print('Val F1: %0.6f' % f1, flush=True)

            if f1 > maxVal:
                maxVal = f1
                print('New Validation F1 Maximum', flush=True)
                if args.model_output_file_prefix is not None:
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()
                    }, args.model_output_file_prefix + '_val_max.tar')

        if args.test_while_training:
            p, r, f1 = evaluate_model(model, test_x, labelsTest, args.batch_size, device)
            print('Test Precision: %0.6f' % p)
            print('Test Recall: %0.6f' % r)
            print('Test F1: %0.6f' % f1, flush=True)

##########################################################################################
# Save the Model
##########################################################################################

if args.model_output_file_prefix is not None:
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, args.model_output_file_prefix + '_epoch_final.tar')