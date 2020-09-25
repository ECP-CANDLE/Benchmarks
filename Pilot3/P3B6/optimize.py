import torch
import candle
import p3b6 as bmk

import numpy as np
import horovod.torch as hvd

import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from sklearn.metrics import f1_score
from mimic_synthetic_data import MimicDatasetSynthetic

from bert import HiBERT


hvd.init()


def initialize_parameters():
    """ Initialize the parameters for the P3B5 benchmark """

    p3b5_bench = bmk.BenchmarkP3B5(
        bmk.file_path,
        "default_model.txt",
        "pytorch",
        prog="p3b5_baseline",
        desc="Differentiable Architecture Search - Pilot 3 Benchmark 5",
    )

    # Initialize parameters
    gParameters = candle.finalize_parameters(p3b5_bench)
    return gParameters


def load_data(gParameters):
    """ Initialize random data

    Args:
        gParameters: parameters from candle

    Returns:
        train, valid, test sets
    """
    num_classes = gParameters["num_classes"]
    num_train_samples = gParameters["num_train_samples"]
    num_valid_samples = gParameters["num_valid_samples"]
    num_test_samples = gParameters["num_test_samples"]

    train = MimicDatasetSynthetic(num_train_samples, num_classes)
    valid = MimicDatasetSynthetic(num_valid_samples, num_classes)
    test = MimicDatasetSynthetic(num_test_samples, num_classes)

    return train, valid, test


def create_data_loaders(gParameters):
    """ Initialize data loaders

    Args:
        gParameters: parameters from candle

    Returns:
        train, valid, test data loaders
    """
    train, valid, test = load_data(gParameters)

    train_sampler = DistributedSampler(
        train, num_replicas=hvd.size(), rank=hvd.rank(), shuffle=True
    )
    valid_sampler = DistributedSampler(
        valid, num_replicas=hvd.size(), rank=hvd.rank(), shuffle=False
    )
    test_sampler = DistributedSampler(
        test, num_replicas=hvd.size(), rank=hvd.rank(), shuffle=False
    )

    batch_size = gParameters["batch_size"]
    train_loader = DataLoader(train, batch_size=batch_size, sampler=train_sampler)
    valid_loader = DataLoader(valid, batch_size=batch_size, sampler=val_sampler)
    test_loader = DataLoader(test, batch_size=batch_size, sampler=test_sampler)

    return train_loader, valid_loader, test_loader


def train(dataloader, model, optimizer, criterion, args, epoch):
    model.train()
    train_sampler.set_epoch(epoch)

    for idx, batch in enumerate(dataloader):
        train_loss = 0.0
        optimizer.zero_grad()

        input_ids = batch["tokens"].to(device)
        segment_ids = batch["seg_ids"].to(device)
        input_mask = batch["masks"].to(device)
        n_segs = batch["n_segs"].to(device)

        logits = model(input_ids, input_mask, segment_ids, n_segs)

        label_ids = batch["label"].to(device)

        loss = criterion(
            logits.view(-1, num_classes), label_ids.view(-1, args.num_classes)
        )

        loss.backward()
        optimizer.step()

        train_loss += loss.mean()

        # track training loss
        if (idx + 1) % 100 == 0:
            train_loss = torch.tensor(train_loss)
            avg_loss = hvd.allreduce(train_loss, name="avg_loss").item()

            if hvd.rank() == 0:
                print(f"epoch: {epoch}, batch: {idx}, loss: {train_loss}")


def validate(dataloader, model, args, epoch):
    model.eval()

    val_preds = []
    val_labels = []

    with torch.no_grad():
        for idx, batch in enumerate(dataloader):

            input_ids = batch["tokens"].to(device)
            segment_ids = batch["seg_ids"].to(device)
            input_mask = batch["masks"].to(device)
            n_segs = batch["n_segs"].to(device)

            logits = model(input_ids, input_mask, segment_ids, n_segs)
            logits = torch.nn.Sigmoid()(logits)

            logits = logits.view(-1, args.num_classes).cpu().data.numpy()
            preds.append(np.rint(logits))
            labels.append(batch["label"].data.numpy())

    preds = np.concatenate(preds, 0)
    labels = np.concatenate(labels, 0)

    preds = torch.tensor(preds)
    preds_all = hvd.allgather(preds, name="val_preds_all").cpu().data.numpy()

    labels = torch.tensor(labels)
    labels_all = hvd.allgather(labels, name="val_labels_all").cpu().data.numpy()

    valid_f1 = f1_score(val_labels_all.flatten(), val_preds_all.flatten())

    if hvd.rank() == 0:
        print(f"epoch: {epoch}, validation F1: {valid_f1}")


def run(params):
    args = candle.ArgumentStruct(**params)
    args.cuda = torch.cuda.is_available()
    device = torch.device(f"cuda" if args.cuda else "cpu")

    train_loader, valid_loader, test_loader = create_data_loaders(params)

    model = model = HiBERT(args.pretrained_weights_path, args.num_classes)
    model.to(device)

    params = [
        {
            "params": [p for n, p in model.named_parameters()],
            "weight_decay": args.weight_decay,
        }
    ]

    optimizer = torch.optim.Adam(params, lr=args.learning_rate, eps=args.eps)
    optimizer = hvd.DistributedOptimizer(
        optimizer, named_parameters=model.named_parameters()
    )

    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(args.num_epochs):
        train(train_loader, model, optimizer, criterion, args, epoch)
        validate(valid_loader, model, args, epoch)


def main():
    params = initialize_parameters()
    run(params)


if __name__ == "__main__":
    main()
