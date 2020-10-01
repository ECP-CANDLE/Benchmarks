"""
    File Name:          unoMT_baseline_pytorch.py
    File Description:   This has been taken from the unoMT original
                        scripts (written by Xiaotian Duan, xduan7@uchicago.edu)
                        and has been modified to fit CANDLE framework.
                        Date: 3/12/19.
"""

import datetime
import numpy as np

from unoMT_pytorch_model import UnoMTModel
import unoMT

import torch

from utils.miscellaneous.random_seeding import seed_random_state

import candle

np.set_printoptions(precision=4)


def initialize_parameters(default_model='unoMT_default_model.txt'):

    # Build benchmark object
    unoMTb = unoMT.unoMTBk(unoMT.file_path, default_model, 'pytorch',
                           prog='unoMT_baseline', desc='Multi-task combined single and combo drug prediction for cross-study data - Pilot 1')

    print("Created unoMT benchmark")

    # Initialize parameters
    gParameters = candle.finalize_parameters(unoMTb)
    print("Parameters initialized")

    return gParameters


def run(params):

    args = candle.ArgumentStruct(**params)
    seed = args.rng_seed
    candle.set_seed(seed)
    # Setting up random seed for reproducible and deterministic results
    seed_random_state(args.rng_seed)

    # check for sufficient number of epochs to start validation
    if params['epochs'] < params['resp_val_start_epoch']:
        raise Exception('Number of epochs is less than validation threshold (resp_val_start_epoch)')

    # Construct extension to save validation results
    now = datetime.datetime.now()
    ext = '%02d%02d_%02d%02d_pytorch' \
        % (now.month, now.day, now.hour, now.minute)

    candle.verify_path(params['save_path'])
    prefix = '{}{}'.format(params['save_path'], ext)
    logfile = params['logfile'] if params['logfile'] else prefix + '.log'
    candle.set_up_logger(logfile, unoMT.logger, params['verbose'])
    unoMT.logger.info('Params: {}'.format(params))

    # Computation device config (cuda or cpu)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    modelUno = UnoMTModel(args, use_cuda, device)

    modelUno.pre_train_config()
    modelUno.train()
    modelUno.print_final_stats()


def main():
    params = initialize_parameters()
    run(params)


if __name__ == '__main__':
    main()
