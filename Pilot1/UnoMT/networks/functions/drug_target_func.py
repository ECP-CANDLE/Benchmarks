"""
    File Name:          UnoPytorch/drug_target_func.py
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               9/4/18
    Python Version:     3.6.6
    File Description:

"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def train_drug_target(device: torch.device,

                      drug_target_net: nn.Module,
                      data_loader: torch.utils.data.DataLoader,

                      max_num_batches: int,
                      optimizer: torch.optim, ):

    drug_target_net.train()

    for batch_idx, (drug_feature, target) in enumerate(data_loader):

        if batch_idx >= max_num_batches:
            break

        drug_feature, target = drug_feature.to(device), target.to(device)

        drug_target_net.zero_grad()
        out_target = drug_target_net(drug_feature)
        F.nll_loss(input=out_target, target=target).backward()
        optimizer.step()


def valid_drug_target(device: torch.device,

                      drug_target_net: nn.Module,
                      data_loader: torch.utils.data.DataLoader, ):

    drug_target_net.eval()

    correct_target = 0

    with torch.no_grad():
        for drug_feature, target in data_loader:

            drug_feature, target = drug_feature.to(device), target.to(device)

            out_target = drug_target_net(drug_feature)
            pred_target = out_target.max(1, keepdim=True)[1]

            correct_target += pred_target.eq(
                target.view_as(pred_target)).sum().item()

    # Get overall accuracy
    target_acc = 100. * correct_target / len(data_loader.dataset)

    print('\tDrug Target Family Classification Accuracy: %5.2f%%' % target_acc)

    return target_acc
