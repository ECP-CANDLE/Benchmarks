"""
    File Name:          UnoPytorch/cl_clf_func.py
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               9/4/18
    Python Version:     3.6.6
    File Description:

"""
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F


def train_cl_clf(device: torch.device,

                 category_clf_net: nn.Module,
                 site_clf_net: nn.Module,
                 type_clf_net: nn.Module,
                 data_loader: torch.utils.data.DataLoader,

                 max_num_batches: int,
                 optimizer: torch.optim, ):

    category_clf_net.train()
    site_clf_net.train()
    type_clf_net.train()

    for batch_idx, (rnaseq, data_src, cl_site, cl_type, cl_category) \
            in enumerate(data_loader):

        if batch_idx >= max_num_batches:
            break

        rnaseq, data_src, cl_site, cl_type, cl_category = \
            rnaseq.to(device), data_src.to(device), cl_site.to(device), \
            cl_type.to(device), cl_category.to(device)

        category_clf_net.zero_grad()
        site_clf_net.zero_grad()
        type_clf_net.zero_grad()

        out_category = category_clf_net(rnaseq, data_src)
        out_site = site_clf_net(rnaseq, data_src)
        out_type = type_clf_net(rnaseq, data_src)

        F.nll_loss(input=out_category, target=cl_category).backward()
        F.nll_loss(input=out_site, target=cl_site).backward()
        F.nll_loss(input=out_type, target=cl_type).backward()

        optimizer.step()


def valid_cl_clf(
        device: torch.device,

        category_clf_net: nn.Module,
        site_clf_net: nn.Module,
        type_clf_net: nn.Module,
        data_loader: torch.utils.data.DataLoader, ):

    category_clf_net.eval()
    site_clf_net.eval()
    type_clf_net.eval()

    correct_category = 0
    correct_site = 0
    correct_type = 0

    with torch.no_grad():
        for rnaseq, data_src, cl_site, cl_type, cl_category in data_loader:

            rnaseq, data_src, cl_site, cl_type, cl_category = \
                rnaseq.to(device), data_src.to(device), cl_site.to(device), \
                cl_type.to(device), cl_category.to(device)

            out_category = category_clf_net(rnaseq, data_src)
            out_site = site_clf_net(rnaseq, data_src)
            out_type = type_clf_net(rnaseq, data_src)

            pred_category = out_category.max(1, keepdim=True)[1]
            pred_site = out_site.max(1, keepdim=True)[1]
            pred_type = out_type.max(1, keepdim=True)[1]

            correct_category += pred_category.eq(
                cl_category.view_as(pred_category)).sum().item()
            correct_site += pred_site.eq(
                cl_site.view_as(pred_site)).sum().item()
            correct_type += pred_type.eq(
                cl_type.view_as(pred_type)).sum().item()

    # Get overall accuracy
    category_acc = 100. * correct_category / len(data_loader.dataset)
    site_acc = 100. * correct_site / len(data_loader.dataset)
    type_acc = 100. * correct_type / len(data_loader.dataset)

    print('\tCell Line Classification: '
          '\n\t\tCategory Accuracy: \t\t%5.2f%%; '
          '\n\t\tSite Accuracy: \t\t\t%5.2f%%; '
          '\n\t\tType Accuracy: \t\t\t%5.2f%%'
          % (category_acc, site_acc, type_acc))

    return category_acc, site_acc, type_acc
