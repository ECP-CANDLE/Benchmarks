import p3b6 as bmk
import candle
import darts


def initialize_parameters():
    """ Initialize the parameters for the P3B5 benchmark """

    p3b5_bench = bmk.BenchmarkP3B5(
        bmk.file_path,
        'default_model.txt',
        'pytorch',
        prog='p3b5_baseline',
        desc='Differentiable Architecture Search - Pilot 3 Benchmark 5',
    )

    # Initialize parameters
    gParameters = candle.finalize_parameters(p3b5_bench)
    #bmk.logger.info('Params: {}'.format(gParameters))
    return gParameters


def fetch_data(gParameters):
    """ Download and untar data

    Args:
        gParameters: parameters from candle

    Returns:
        path to where the data is located
    """
    path = gParameters['data_url']
    fpath = candle.fetch_file(path + gParameters['train_data'], 'Pilot3', untar=True)
    return fpath


def run(params):
    args = candle.ArgumentStruct(**params)
    args.cuda = torch.cuda.is_available()

    device = torch.device(f'cuda' if args.cuda else "cpu")


def main():
    params = initialize_parameters()
    run(params)


if __name__=='__main__':
    main()
