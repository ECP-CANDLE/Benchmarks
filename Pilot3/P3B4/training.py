import pandas as pd
import numpy as np
import os
import time
import sys
import json
import random

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torch import nn
import torch.nn.functional as F
from sklearn.metrics import f1_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

########################################
#
#   define functions to run the model
#
########################################

def get_keyword_tensors(tasks,id2word,id2label):
    Xs = {}
    Ys = {}
    word2id = {v:k for k,v in id2word.items()}
    for t,task in enumerate(tasks):
        kw = json.load(open('./tools/keywords/%s_cuis.json' % task,'r'))
        label2id = {v:k for k,v in id2label[task].items()}
        X = []
        Y = []
        for label,word_lists in kw.items():
            if label not in label2id:
                continue

            keyword_id_lists = []
            for word_list in word_lists:
                id_list = [word2id[w] for w in word_list if w in word2id]
                if len(id_list) == len(word_list):
                    keyword_id_lists.append(id_list)

            if len(keyword_id_lists) > 0:
                Y.append(label2id[label])
                X.append(keyword_id_lists)

        Xs[task] = X
        Ys[task] = Y
    return Xs,Ys

def keywords_loss(model,tasks,Xs,Ys,max_len,alpha=1.0,n=128,k=5):
    loss = 0
    loss_fct = torch.nn.CrossEntropyLoss()
    for t,task in enumerate(tasks):
        X_ = Xs[task]
        Y = Ys[task]
        XY = list(zip(X_,Y))
        random.shuffle(XY)
        X_,Y = zip(*XY)
        X_ = X_[:n]
        Y = Y[:n]
        X = np.zeros((len(X_),max_len), dtype=int)
        for i,x in enumerate(X_):
            random.shuffle(x)
            flattened_x = np.concatenate(x[:k]).astype(int)
            keyword_len = min(max_len, len(flattened_x))
            X[i,:keyword_len] = flattened_x[:keyword_len]
        X = torch.tensor(X,dtype=torch.long).to(device)
        Y = torch.tensor(Y,dtype=torch.long).to(device)
        logits = model(X)[t]
        loss += loss_fct(logits,Y)
    loss /= len(tasks)
    loss *= alpha
    return loss

def abstain_loss(loss_fct,
                 y_pred,
                 y_true,
                 alpha,
                 ntask_abs_prob=1,
                 ):

    # calculate cross entropy on real classes
    y_pred1 = y_pred[:, :-1]
    h_c = loss_fct(y_pred1, y_true)

    # account for multilabel sigmoid cross entropy if used
    if len(h_c.shape) == 2:
        h_c = torch.mean(h_c,-1)
        p_extra = torch.sigmoid(y_pred)[:, -1] - 1e-6
    else:
        p_extra = F.softmax(y_pred, -1)[:, -1] - 1e-6

    # normalize cross entropy loss with abstain probability
    h_c = ((1 - p_extra) + (1 - ntask_abs_prob)) * h_c

    # include the alpha weight term in the loss
    loss = h_c - alpha * torch.log(1 - p_extra)

    return torch.mean(loss)


def train(model,
          optimizer,
          tasks,
          train_loader,
          val_loader=None,
          epochs=100,
          patience=5,
          savepath=None,
          class_weights=None,
          multilabel=False,
          keywords=False,
          id2label=None,
          id2word=None,
          abstain_args={}
          ):

    ntask_flag = False
    abstain_flag = False

    # setup loss function
    class_weights_tensor = None
    if class_weights is not None:
        class_weights_tensor = torch.FloatTensor(class_weights).to(device)
    if multilabel:
        loss_fct = torch.nn.BCEWithLogitsLoss(class_weights_tensor)
    else:
        loss_fct = torch.nn.CrossEntropyLoss(class_weights_tensor)
    num_tasks = len(tasks)

    # setup keywords
    if keywords:
        kw_X, kw_Y = get_keyword_tensors(tasks,id2word,id2label)

    # vars for tracking patience
    bestloss = np.inf
    pat_counter = 0
    use_amp = False
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # train model
    for ep in range(epochs):

        model.train()
        y_preds = [[] for c in range(num_tasks)]
        y_trues = [[] for c in range(num_tasks)]
        ntask_filter = []
        start_time = time.time()

        for b, batch in enumerate(train_loader):
            with torch.cuda.amp.autocast(enabled=use_amp):
                optimizer.zero_grad()
                X = batch['X'].to(device)
                ys = [batch['y_%s' % t].to(device) for t in tasks]
                logits = model(X)
                loss = 0

                # if ntask abstain is enabled, get ntask abstain probability
                ntask_abs_prob = 1
                if ntask_flag:
                    ntask_abs_prob = torch.sigmoid(logits[-1])[:, -1]
                    ntask_filter_ = np.rint(ntask_abs_prob.detach().cpu().numpy())
                    ntask_filter.extend(~ntask_filter_.astype(np.bool))

                for t, task in enumerate(tasks):

                    # track predicted and true labels
                    if multilabel:
                        y_trues[t].extend(np.argmax(ys[t].detach().cpu().numpy(), 1))
                    else:
                        y_trues[t].extend(ys[t].detach().cpu().numpy())
                    y_preds[t].extend(np.argmax(logits[t].detach().cpu().numpy(), 1))

                    # optional abstain loss
                    if abstain_flag:
                        if task in ntask_tasks:
                            filtered_ntask_abs = ntask_abs_prob
                        else:
                            filtered_ntask_abs = 1
                        loss += abstain_loss(
                                             loss_fct,
                                             logits[t],
                                             ys[t],
                                             abstain_alphas[t],
                                             filtered_ntask_abs,
                                            )
                    else:
                        loss += loss_fct(logits[t], ys[t])

                # add optional ntask abstain loss
                if ntask_flag:
                    loss -= torch.mean(ntask_alpha * torch.log(1 - ntask_abs_prob + 1e-6))
                loss /= len(tasks)

                # add optional keyword loss
                if keywords:
                    max_len = train_loader.dataset.max_len
                    loss += keywords_loss(model,tasks,kw_X,kw_Y,max_len)

            # backprop
            scaler.scale(loss).backward() # loss.backward()
            scaler.step(optimizer) # optimizer.step()
            scaler.update()
            optimizer.zero_grad()
            l = loss.cpu().detach().numpy()
            sys.stdout.write("epoch %i, sample %i of %i, loss: %f           \r"\
                             % (ep + 1, (b + 1) * train_loader.batch_size, len(train_loader.dataset), l))
            sys.stdout.flush()

        print()
        print("\ntraining time: %.2f" % (time.time() - start_time))
        print_header(abstain_flag)

        # track train scores for abstention in case no validation set
        micros = []
        abs_rates = []

        for t, task in enumerate(tasks):

            y_trues_ = y_trues[t]
            y_preds_ = y_preds[t]

            # calculate abstention if enabled
            if abstain_flag:
                predicted_idx = np.array(y_preds[t]) != abstain_labels[t]
                abs_rate = 1 - np.sum(predicted_idx) / len(predicted_idx)
                abs_rates.append(abs_rate)
                y_trues_ = np.array(y_trues[t])[predicted_idx]
                y_preds_ = np.array(y_preds[t])[predicted_idx]

                # task specific pruning for ntask filter
                if ntask_flag and task in ntask_tasks:
                    ntask_filter = np.logical_and(ntask_filter, predicted_idx)

            # calculate micro and macro
            micro = f1_score(y_trues_, y_preds_, average='micro')
            macro = f1_score(y_trues_, y_preds_, average='macro')
            micros.append(micro)
            if abstain_flag:
                print_stats(task, micro, macro, abs_rate)
            else:
                print_stats(task, micro, macro)

        # calculate optional ntask accuracy
        if ntask_flag:
            ntask_acc = np.ones_like(y_trues[0])[ntask_filter]
            for t, task in enumerate(tasks):
                if task in ntask_tasks:
                    y_trues_ = np.array(y_trues[t])[ntask_filter]
                    y_preds_ = np.array(y_preds[t])[ntask_filter]
                    correct = y_trues_ == y_preds_
                    ntask_acc = np.logical_and(ntask_acc, correct)
            ntask_acc = np.sum(ntask_acc) / len(ntask_acc)
            ntask_abs_rate = 1 - np.sum(ntask_filter) / len(ntask_filter)
            print_stats('ntask', ntask_acc, ntask_acc, ntask_abs_rate)

        # check validation performance
        if val_loader is not None:
            base_scores, abs_scores = score(
                                     model,
                                     tasks,
                                     val_loader,
                                     class_weights,
                                     multilabel,
                                     abstain_args
                                    )

            # compute alpha scale factors and individual stop metrics
            if abstain_flag:
                task_scale_factors, task_stop_metrics = modify_alphas(
                             abstain_alphas,
                             base_scores['micros'],
                             abs_scores['abs_rates'],
                             abstain_min_acc,
                             abstain_max_abs,
                             abstain_abs_gain,
                             abstain_acc_gain,
                             abstain_alpha_scale,
                             abstain_tune_mode,
                             additive=True,
                            )
                if ntask_flag:
                    ntask_scale_factor, ntask_stop_metric = modify_alphas(
                              [ntask_alpha],
                              [abs_scores['ntask_acc']],
                              [abs_scores['ntask_abs_rate']],
                              [ntask_min_acc],
                              [ntask_max_abs],
                              abstain_abs_gain,
                              abstain_acc_gain,
                              abstain_alpha_scale,
                              abstain_tune_mode,
                              additive=True,
                             )
                    ntask_scale_factor = ntask_scale_factor[0]
                    ntask_stop_metric = ntask_stop_metric[0]

            print("epoch %i validation" % (ep + 1))
            if abstain_flag:
                print_abs_tune_header(abstain_tune_mode)
            else:
                print_header(abstain_flag)

            # arrays to store all values including ntask
            stop_metrics = []
            all_accs = []
            all_abs = []
            all_alphas = []

            micros = base_scores['micros']
            macros = base_scores['macros']
            val_loss = base_scores['val_loss']
            abs_rates = abs_scores['abs_rates']

            for t, task in enumerate(tasks):

                if abstain_flag:
                    print_abs_tune_stats(abstain_tune_mode, task,
                                    macros[t], micros[t], abstain_min_acc[t],
                                    abs_rates[t], abstain_max_abs[t],
                                    abstain_alphas[t], task_scale_factors[t], task_stop_metrics[t])

                    # build stop_metrics list
                    stop_metrics.append(task_stop_metrics[t])
                    all_accs.append(micros[t])
                    all_abs.append(abs_rates[t])
                    all_alphas.append(abstain_alphas[t])
                else:
                    print_stats(task, micros[t], macros[t])

            ntask_acc = abs_scores['ntask_acc']
            ntask_abs_rate = abs_scores['ntask_abs_rate']

            if ntask_flag:
                print_abs_tune_stats(abstain_tune_mode, 'ntask',
                                ntask_acc, ntask_acc, ntask_min_acc,
                                ntask_abs_rate, ntask_max_abs,
                                ntask_alpha, ntask_scale_factor, ntask_stop_metric)

                # build stop_metrics list
                stop_metrics.append(ntask_stop_metric)
                all_accs.append(ntask_acc)
                all_abs.append(ntask_abs_rate)
                all_alphas.append(ntask_alpha)

            if abstain_flag:
                write_abs_stats(all_alphas, all_accs, all_abs, stop_metrics, savepath)

            stop_norm = 'max'
            # track best validation loss, save best model, track patience
            if abstain_flag:
                # use the stop_metric for stopping
                print('epoch %i' % (ep + 1))
                if stop_norm == 'max':
                    # max value
                    stop_metric = np.linalg.norm(np.array(stop_metrics), np.inf)
                elif stop_norm == 'l2':
                    # l2 norm
                    stop_metric = np.linalg.norm(np.array(stop_metrics))

                torch.save(model.state_dict(), savepath)
                torch.save(model.state_dict(), savepath + '%i' % ep)
                if (stop_metric < abstain_stop_limit):
                    print('Stopping criterion reached: %.4f < %.4f' % (stop_metric, abstain_stop_limit))
                    break
                else:
                    print('Stopping criterion not reached: %.4f > %.4f' % (stop_metric, abstain_stop_limit))
                #checkpoint at every epoch until stopping criteria reached

            else:
                print("epoch %i val loss: %.8f, best val loss: %.8f" % (ep + 1, val_loss, bestloss))
                # use patience based on val_loss
                if val_loss < bestloss:
                    bestloss = val_loss
                    pat_counter = 0
                    torch.save(model.state_dict(), savepath)
                else:
                    pat_counter += 1
                    if pat_counter >= patience:
                        break
                print('patience counter is at %i of %i' % (pat_counter, patience))

        # if no validation set, save after every epoch
        else:
            torch.save(model.state_dict(), savepath)

        # update abstain alphas
        if abstain_flag:
            for i, alpha in enumerate(abstain_alphas):
                abstain_alphas[i] = abstain_alphas[i] * task_scale_factors[i]
            print('New alphas', ['%.4f' % a for a in abstain_alphas])
            if ntask_flag:
                ntask_alpha = ntask_alpha * ntask_scale_factor
                print('New ntask alpha', ['%.4f' % ntask_alpha])


def score(model,
          tasks,
          data_loader,
          class_weights=None,
          multilabel=False,
          abstain_args = {}
         ):

    # set up dictionaries for return values
    base_scores = {}
    abs_scores = {}

    # abstain loss requires additional operations before averaging over batch
    reduction = 'mean'
    ntask_flag = False
    abstain_flag = False

    # setup loss function
    class_weights_tensor = None
    if class_weights is not None:
        class_weights_tensor = torch.FloatTensor(class_weights).to(device)
    if multilabel:
        loss_fct = torch.nn.BCEWithLogitsLoss(class_weights_tensor, reduction=reduction)
    else:
        loss_fct = torch.nn.CrossEntropyLoss(class_weights_tensor, reduction=reduction)
    num_tasks = len(tasks)

    # evaluate
    model.eval()
    y_preds = [[] for c in range(num_tasks)]
    y_trues = [[] for c in range(num_tasks)]
    ntask_filter = []
    losses = []

    with torch.no_grad():
        for b, batch in enumerate(data_loader):
            X = batch['X'].to(device)
            ys = [batch['y_%s' % t].to(device) for t in tasks]
            logits = model(X)
            loss = 0

            # if ntask abstain is enabled, get ntask abstain rate
            ntask_abs_prob = 1
            if ntask_flag:
                ntask_abs_prob = torch.sigmoid(logits[-1])[:, -1]
                ntask_filter_ = np.rint(ntask_abs_prob.detach().cpu().numpy())
                ntask_filter.extend(~ntask_filter_.astype(np.bool))

            for t, task in enumerate(tasks):

                # track predicted and true labels
                if multilabel:
                    y_trues[t].extend(np.argmax(ys[t].detach().cpu().numpy(), 1))
                else:
                    y_trues[t].extend(ys[t].detach().cpu().numpy())
                y_preds[t].extend(np.argmax(logits[t].detach().cpu().numpy(), 1))

                # optional abstain loss
                if abstain_flag:
                    if task in ntask_tasks:
                        filtered_ntask_abs = ntask_abs_prob
                    else:
                        filtered_ntask_abs = 1
                    loss += abstain_loss(
                                         loss_fct,
                                         logits[t],
                                         ys[t],
                                         abstain_alphas[t],
                                         filtered_ntask_abs,
                                        )
                else:
                    loss += loss_fct(logits[t], ys[t])

            # add optional ntask abstain loss
            if ntask_flag:
                loss -= torch.mean(ntask_alpha * torch.log(1 - ntask_abs_prob))

            loss /= len(tasks)
            l = loss.cpu().detach().numpy()
            losses.append(l)
            sys.stdout.write("predicting sample %i of %i           \r"\
                             % ((b + 1) * data_loader.batch_size, len(data_loader.dataset)))
            sys.stdout.flush()

    # get scores
    print()
    micros = []
    macros = []
    abs_rates = []
    for t, task in enumerate(tasks):

        # calculate abstention if enabled

        y_trues_ = y_trues[t]
        y_preds_ = y_preds[t]

        # calculate abstention if enabled
        if abstain_flag:
            predicted_idx = np.array(y_preds[t]) != abstain_labels[t]
            abs_rate = 1 - np.sum(predicted_idx) / len(predicted_idx)
            abs_rates.append(abs_rate)
            y_trues_ = np.array(y_trues[t])[predicted_idx]
            y_preds_ = np.array(y_preds[t])[predicted_idx]

            # task specific pruning for ntask filter
            if ntask_flag and task in ntask_tasks:
                ntask_filter = np.logical_and(ntask_filter, predicted_idx)

        # calculate micro and macro
        micro = f1_score(y_trues_, y_preds_, average='micro')
        macro = f1_score(y_trues_, y_preds_, average='macro')
        micros.append(micro)
        macros.append(macro)

    # calculate optional ntask accuracy
    ntask_acc = 1
    ntask_abs_rate = 0
    if ntask_flag:
        ntask_acc = np.ones_like(y_trues[0])[ntask_filter]
        for t, task in enumerate(tasks):
            if task in ntask_tasks:
                y_trues_ = np.array(y_trues[t])[ntask_filter]
                y_preds_ = np.array(y_preds[t])[ntask_filter]
                correct = y_trues_ == y_preds_
                ntask_acc = np.logical_and(ntask_acc, correct)

        if (len(ntask_acc) > 0):
            ntask_acc = np.sum(ntask_acc) / len(ntask_acc)
        else:
            ntask_acc = 1.0
        ntask_abs_rate = 1 - np.sum(ntask_filter) / len(ntask_filter)

    base_scores['micros'] = micros
    base_scores['macros'] = macros
    base_scores['val_loss'] = np.mean(losses)

    abs_scores['abs_rates'] = abs_rates
    abs_scores['ntask_acc'] = ntask_acc
    abs_scores['ntask_abs_rate'] = ntask_abs_rate

    #return micros, macros, np.mean(losses), abs_rates, ntask_acc, ntask_abs_rate
    return base_scores, abs_scores


def predict(model,
            tasks,
            data_loader,
            multilabel=False,
            ntask=False,
            ):

    num_tasks = len(tasks)
    model.eval()
    y_preds = [[] for c in range(num_tasks)]
    y_probs = [[] for c in range(num_tasks)]
    ntask_abs_probs = []
    with torch.no_grad():
        for b, batch in enumerate(data_loader):
            X = batch['X'].to(device)
            logits = model(X)
            for t, task in enumerate(tasks):
                y_preds[t].extend(np.argmax(logits[t].detach().cpu().numpy(), 1))
                if multilabel:
                    y_prob = torch.sigmoid(logits[t]).detach().cpu().numpy()
                else:
                    y_prob = F.softmax(logits[t],-1).detach().cpu().numpy()
                y_probs[t].append(y_prob)
            if ntask:
                ntask_abs_prob = torch.sigmoid(logits[-1])[:,-1].detach().cpu().numpy()
                ntask_abs_probs.extend(ntask_abs_prob)
            sys.stdout.write("predicting sample %i of %i           \r"\
                             % ((b + 1) * data_loader.batch_size, len(data_loader.dataset)))
            sys.stdout.flush()
    print()
    y_probs = [np.vstack(y_probs[t]) for t in range(len(tasks))]
    return y_preds, y_probs, ntask_abs_probs


def modify_alphas(alphas,
                  micros,
                  abs_rates,
                  min_acc,
                  max_abs,
                  abs_gain,
                  acc_gain,
                  alpha_scale,
                  tune_mode,
                  additive=True):

    #new_alphas = []
    scale_factors = []
    stop_metrics = []

    for i, alpha in enumerate(alphas):

        # these are common to all tuning methods
        acc_error = micros[i] - min_acc[i]
        abs_error = abs_rates[i] - max_abs[i]
        acc_ratio = micros[i] / min_acc[i]
        abs_ratio = abs_rates[i] / max_abs[i]

        min_scale = alpha_scale[i]
        max_scale = 1 / min_scale

        if (tune_mode == 'abs_acc'):
            # modify the scaling factor according to error in target abstention and accuracy
            #print('Tuning for abstention and accuracy')

            # clip if accuracy is above min_acc
            acc_error = min([acc_error, 0.0])

            # clip if abstention is below max_abs
            abs_error = max([abs_error, 0.0])

            # multiplicative scaling
            # clip if accuracy is above min_acc
            acc_ratio = min([acc_ratio, 1.0])

            # clip if abstention is below max_abs
            abs_ratio = max([abs_ratio, 1.0])

            # choose multiplicative or additive scaling
            if additive:
                new_scale = (1. + acc_gain * acc_error + abs_gain * abs_error)
            else:
                new_scale = (1. * acc_ratio * abs_ratio)

            # use harmonic mean to rescale the stopping criterion
            stop_i = (new_scale - 1.) * ((1. / acc_gain) + (1. / abs_gain)) * 0.5

        elif (tune_mode == 'acc'):

            #print('Tuning for accuracy')
            # no clipping here
            new_scale = (1. + acc_gain * acc_error)
            stop_i = acc_error
            # special case of there is no abstention and accuracy is more than requested
            # to avoid this task preventing stopping we set the stop metric to zero.
            if (acc_error > 0.0) and (abs_rates[i] < 1e-8):
                stop_i = 0.0


        elif (tune_mode == 'abs'):

            #print('Tuning for abstention')
            # no clipping here
            new_scale = (1. + abs_gain * abs_error)
            stop_i = abs_error

        # this is common for all scaling modes
        # threshold the scaling to be safe
        new_scale = min([new_scale, max_scale])
        new_scale = max([new_scale, min_scale])

        #new_alphas.append(alphas[i] * new_scale)
        scale_factors.append(new_scale)
        stop_metrics.append(stop_i)

    return scale_factors, stop_metrics


def write_abs_header(tasks, ntask, savepath):
    path = os.path.join('predictions',".".join(os.path.basename(savepath).split(".")[:-1]))
    path += '_abs_stats.txt'
    abs_file = open(path, 'w+')
    abs_file.write("Alphas, accuracies, abstention, stop_metric\n")
    # we just want 4 copies on the header line
    for n in range(4):
        for i, task in enumerate((tasks)):
            abs_file.write("%11s " % task)
        if ntask:
            abs_file.write("%11s " % 'ntask')

    abs_file.write("\n")  # this should produce a newline
    abs_file.close()


def write_abs_stats(alphas, accs, abs_frac, stop_metrics, savepath):
    path = os.path.join('predictions',".".join(os.path.basename(savepath).split(".")[:-1]))
    path += '_abs_stats.txt'
    abs_file = open(path, 'a')
    # write a single line with alphas, accuracy and abstention
    for i in range(len(alphas)):
        abs_file.write("%10.5e " % alphas[i])
    for i in range(len(alphas)):
        abs_file.write("%10.5e " % accs[i])
    for i in range(len(alphas)):
        abs_file.write("%10.5e " % abs_frac[i])
    for i in range(len(alphas)):
        abs_file.write("%10.5e " % stop_metrics[i])
    abs_file.write("\n")  # this should produce a newline

    abs_file.close()


def print_abs_tune_header(tune_mode):
    # change the output based on tuning mode
    if (tune_mode == 'abs_acc'):
        print("%12s, %10s, %10s, %10s, %10s, %10s, %10s, %10s, %10s" % \
              ('task', 'micro', 'macro', 'abs_frac', 'min_acc', 'max_abs', 'alpha', 'scale_fac', 'stop_metric'))
    elif (tune_mode == 'abs'):
        print("%12s, %10s, %10s, %10s, %10s, %10s, %10s, %10s" % \
              ('task', 'micro', 'macro', 'abs_frac', 'max_abs', 'alpha', 'scale_fac', 'stop_metric'))
    elif (tune_mode == 'acc'):
        print("%12s, %10s, %10s, %10s, %10s, %10s, %10s, %10s" % \
              ('task', 'micro', 'macro', 'abs_frac', 'target_acc', 'alpha', 'scale_fac', 'stop_metric'))
    else:
        print("Tuning mode must be one of [abs_acc, abs, acc]")


def print_abs_tune_stats(tune_mode, task, macro, micro, min_acc, abs_frac, max_abs, alpha, scale_fac, stop_metric):
    if (tune_mode == 'abs_acc'):
        print("%12s, %10.4f, %10.4f, %10.4f, %10.4f, %10.4f, %10.4f, %10.4f, %10.4f" %
              (task, micro, macro, abs_frac, min_acc, max_abs, alpha, scale_fac, stop_metric))
    elif (tune_mode == 'abs'):
        print("%12s, %10.4f, %10.4f, %10.4f, %10.4f, %10.4f, %10.4f, %10.4f" %
              (task, micro, macro, abs_frac, max_abs, alpha, scale_fac, stop_metric))
    elif (tune_mode == 'acc'):
        print("%12s, %10.4f, %10.4f, %10.4f, %10.4f, %10.4f, %10.4f, %10.4f" %
              (task, micro, macro, abs_frac, min_acc, alpha, scale_fac, stop_metric))


def print_header(abstain_flag):
    if abstain_flag:
        print("%12s, %10s, %10s, %10s" %
              ('task', 'micro', 'macro', 'abs_frac'))
    else:
        print("%12s, %10s, %10s" %
              ('task', 'micro', 'macro'))


def print_stats(task, micro, macro, abs_frac=None):
    if abs_frac is not None:
        print("%12s, %10.4f, %10.4f, %10.4f" % (task, micro, macro, abs_frac))
    else:
        print("%12s, %10.4f, %10.4f" % (task, micro, macro))


def print_all_stats(tasks, base_scores, abs_scores, abstain_args):

    micros = base_scores['micros']
    macros = base_scores['macros']
    abs_rates = abs_scores['abs_rates']

    abstain_flag = abstain_args['abstain_flag']
    ntask_flag = abstain_args['ntask_flag']

    print_header(abstain_flag)

    for t, task in enumerate(tasks):
        if abstain_flag:
            print_stats(task, micros[t], macros[t], abs_rates[t])
        else:
            print_stats(task, micros[t], macros[t])

    ntask_acc = abs_scores['ntask_acc']
    ntask_abs_rate = abs_scores['ntask_abs_rate']

    if abstain_flag and ntask_flag:
        print_stats('ntask', ntask_acc, ntask_acc, ntask_abs_rate)


def set_abstain_args(model_args, id2label={}):

    tasks = model_args['data_kwargs']['tasks']

    # set up abstain args
    abstain_flag = model_args['abstain_kwargs']['abstain']
    ntask_flag = model_args['abstain_kwargs']['ntask']
    id2label_new = id2label.copy()

    # some values are set/changed from input values
    if abstain_flag:
        abstain_labels = [len(id2label[t].keys()) for t in tasks]
        num_classes = [len(id2label[t].keys()) + 1 for t in tasks]

        # ntask model has an extra task to determine ntask abstain
        if ntask_flag:
            num_classes.append(1)

        # add additional abstain class in id2label
        for task in tasks:
            idx = len(id2label_new[task])
            if id2label_new[task][idx-1] != 'abs_%s' % task:
                id2label_new[task][idx] = 'abs_%s' % task

    else:
        abstain_labels = None
        num_classes = [len(id2label[t].keys()) for t in tasks]

        # to be safe, set ntask to false if abstain is disabled
        ntask_flag = False

    abstain_args = {}
    abstain_args['abstain_flag'] = abstain_flag
    abstain_args['ntask_flag'] = ntask_flag
    abstain_args['abstain_labels'] = abstain_labels
    # rest are directly from input
    abstain_args['abstain_alphas'] = model_args['abstain_kwargs']['alphas']
    abstain_args['abstain_min_acc'] = model_args['abstain_kwargs']['min_acc']
    abstain_args['abstain_max_abs'] = model_args['abstain_kwargs']['max_abs']
    abstain_args['abstain_abs_gain'] = model_args['abstain_kwargs']['abs_gain']
    abstain_args['abstain_acc_gain'] = model_args['abstain_kwargs']['acc_gain']
    abstain_args['abstain_tune_mode'] = model_args['abstain_kwargs']['tune_mode']
    abstain_args['abstain_stop_limit'] = model_args['abstain_kwargs']['stop_limit']
    abstain_args['abstain_alpha_scale'] = model_args['abstain_kwargs']['alpha_scale']
    abstain_args['ntask_alpha'] = model_args['abstain_kwargs']['ntask_alpha']
    abstain_args['ntask_max_abs'] = model_args['abstain_kwargs']['ntask_max_abs']
    abstain_args['ntask_min_acc'] = model_args['abstain_kwargs']['ntask_min_acc']
    abstain_args['ntask_tasks'] = model_args['abstain_kwargs']['ntask_tasks']

    if abstain_flag:
        print('Running with:')
        for key in abstain_args:
            print(key, abstain_args[key])

    return abstain_args, num_classes, id2label_new

