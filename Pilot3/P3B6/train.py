from mimic_synthetic_data import MimicDatasetSynthetic

from torch.utils.data.distributed import DistributedSampler

import candle
import p3b6 as bmk


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
    # bmk.logger.info('Params: {}'.format(gParameters))
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

    train_loader = DataLoader(train, batch_size=1, sampler=train_sampler)
    valid_loader = DataLoader(valid, batch_size=1, sampler=val_sampler)
    test_loader = DataLoader(test, batch_size=1, sampler=test_sampler)

    return train_loader, valid_loader, test_loader


def run(params):
    args = candle.ArgumentStruct(**params)
    args.cuda = torch.cuda.is_available()

    device = torch.device(f"cuda" if args.cuda else "cpu")

    train_loader, valid_loader, test_loader = create_data_loaders(params)


def main():
    params = initialize_parameters()
    run(params)


if __name__ == "__main__":
    main()
