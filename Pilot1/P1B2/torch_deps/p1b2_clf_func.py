import os
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
from pytorch_utils import build_loss


def train_p1b2_clf(device: torch.device,

                 category_clf_net: nn.Module,
                 data_loader: torch.utils.data.DataLoader,
                 loss_type: str,
                 max_num_batches: int,
                 optimizer: torch.optim,
                 scheduler: torch.optim.lr_scheduler,
                 epoch: int, 
                 log_interval: int,
                 dry_run: bool = False, ):

    category_clf_net.train()
    pid = os.getpid()

    correct_category = 0
    train_loss = 0

    for batch_idx, (rnaseq, cl_category) \
            in enumerate(data_loader):

        if batch_idx >= max_num_batches:
            break

        rnaseq, cl_category = \
            rnaseq.to(device),  cl_category.to(device)

        category_clf_net.zero_grad()

        out_category = category_clf_net(rnaseq)

        loss = build_loss(loss_type, out_category, cl_category)
        train_loss += data_loader.batch_size * loss.item() # sum up batch loss
        loss.backward()

        optimizer.step()

        if batch_idx % log_interval == 0:
            print('{}\tTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:5.5f}'.format(
                pid, epoch+1, (batch_idx+1) * len(rnaseq), len(data_loader.dataset),
                100. * (batch_idx+1) / len(data_loader), loss.item()))
            if dry_run:
                break

        
        pred_category = out_category.max(1, keepdim=True)[1]

        correct_category += pred_category.eq(
                cl_category.view_as(pred_category)).sum().item()


    # Get overall accuracy
    train_loss /= len(data_loader.dataset)
    category_acc = 100. * correct_category / len(data_loader.dataset)


    print('\tP1B2 classification: '
          '\n\t\tTraining Loss: \t\t\t%5.5f '
          '\n\t\tTraining Accuracy: \t\t%5.2f%%'
          % (train_loss, category_acc))

    return category_acc, train_loss
    


def valid_p1b2_clf(
        device: torch.device,
        category_clf_net: nn.Module,
        data_loader: torch.utils.data.DataLoader,
        loss_type: str, ):

    category_clf_net.eval()

    correct_category = 0
 
    test_loss = 0

    with torch.no_grad():
        for rnaseq, cl_category in data_loader:

            rnaseq, cl_category = \
                rnaseq.to(device),  cl_category.to(device)

            out_category = category_clf_net(rnaseq)

            loss = build_loss(loss_type, out_category, cl_category)
            test_loss += data_loader.batch_size * loss.item() # sum up batch loss

            pred_category = out_category.max(1, keepdim=True)[1]

            correct_category += pred_category.eq(
                cl_category.view_as(pred_category)).sum().item()


    # Get overall accuracy
    test_loss /= len(data_loader.dataset)
    category_acc = 100. * correct_category / len(data_loader.dataset)


    print('\tP1B2 classification: '
          '\n\t\tValidation Loss: \t\t%5.5f '
          '\n\t\tValidation Accuracy: \t\t%5.2f%%'
          % (test_loss, category_acc))

    return category_acc, test_loss
