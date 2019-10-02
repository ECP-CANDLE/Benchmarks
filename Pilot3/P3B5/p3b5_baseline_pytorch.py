import candle
import p3b5_darts as bmk


def initialize_parameters():
    """ Initialize the parameters for the P3B5 benchmark """

    p3b5_bench = bmk.BenchmarkP3B3(
        bmk.file_path,
        'p3b5_default_model.txt',
        'pytorch',
        prog='p3b5_baseline',
        desc='Differentiable Architecture Search - Pilot 3 Benchmark 5',
    )

    # Initialize parameters
    gParameters = candle.initialize_parameters(p3b5_bench)
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
