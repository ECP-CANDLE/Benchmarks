from __future__ import print_function

import numpy as np
import os
import sys

import datetime

import torch


if True:
    print("Restricting #of GPUs to 8")
    os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3,4,5,6,7"
    #os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15"

file_path = os.path.dirname(os.path.realpath(__file__))
lib_path2 = os.path.abspath(os.path.join(file_path, '..', '..', 'common'))
sys.path.append(lib_path2)
os.chdir(file_path)

import tc1 as bmk
import candle
from torch_deps.tc1_pytorch_model import TC1Model
from torch_deps.random_seeding import seed_random_state

np.set_printoptions(precision=4)

def initialize_parameters(default_model = 'tc1_default_model_pytorch.txt'):

    # Build benchmark object
    tc1Bmk = bmk.BenchmarkTC1(file_path, default_model, 'pytorch',
    prog='tc1_baseline', desc='Multi-task (DNN) for data extraction from clinical reports - Pilot 3 Benchmark 1')

    print("Created Tc1 benchmark")
    
    # Initialize parameters
    gParameters = candle.finalize_parameters(tc1Bmk)
    #benchmark.logger.info('Params: {}'.format(gParameters))
    print("Parameters initialized")

    return gParameters


def run(params):

    args = candle.ArgumentStruct(**params)
    args.no_cuda = args.no_cuda if hasattr(args,'no_cuda') else False
    args.multi_gpu = args.multi_gpu if hasattr(args,'multi_gpu') else True
    args.max_num_batches = args.max_num_batches if hasattr(args,'max_num_batches') else 1000
    args.dry_run = args.dry_run if hasattr(args,'dry_run') else False
    args.log_interval = args.log_interval if hasattr(args,'log_interval') else 8


    seed = args.rng_seed
    candle.set_seed(seed)
    # Setting up random seed for reproducible and deterministic results
    seed_random_state(args.rng_seed)

    # Construct extension to save validation results
    now = datetime.datetime.now()
    ext = '%02d%02d_%02d%02d_pytorch' \
        % (now.month, now.day, now.hour, now.minute)

    candle.verify_path(params['save_path'])
    prefix = '{}{}'.format(params['save_path'], ext)
    logfile = params['logfile'] if params['logfile'] else prefix+'.log'
    candle.set_up_logger(logfile, bmk.logger, params['verbose'])
    bmk.logger.info('Params: {}'.format(params))
    
    args.tensorboard_dir = "tb/{}".format(ext)
    args.logger = bmk.logger 

    #Autosave model 
    model_name = params['model_name']
    args_filename = "{}.model.args".format(params['save_path'])
    args.model_autosave_filename = "{}.autosave.model.pth".format(params['save_path'])
    # CSV logging
    args.csv_filename = '{}{}_training.log'.format(params['save_path'], ext)

    # Computation device config (cuda or cpu)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    # save args to file    
    import pickle
    args_file = open(args_filename, 'wb')
    pickle.dump(args, args_file)
    args_file.close()

    modelTc1 = TC1Model(args, use_cuda, device)
    
    #model.summary()
    #print(modelTc1.tc1_net)  # Model summary
    bmk.logger.info('Model summary: {}'.format(modelTc1.tc1_net))  # Model summary

    modelTc1.train()
    modelTc1.print_final_stats()

    #Save model 
    model_name = params['model_name']
    model_filename = "{}.model_state_dict.pth".format(params['save_path'])
    if hasattr(modelTc1.tc1_net,'module'):
        # Saving the DataParallel model
        torch.save(modelTc1.tc1_net.module.state_dict(), model_filename)
    else:
        torch.save(modelTc1.tc1_net.state_dict(), model_filename)

    #reload args from file
    args_file = open(args_filename, 'rb')
    loaded_args = pickle.load(args_file)
    args_file.close()

    # load weights into new model
    loaded_modelTc1 = TC1Model(loaded_args)
    loaded_modelTc1.tc1_net.load_state_dict(torch.load(model_filename, map_location=torch.device('cpu')))
    print("Loaded torch model from disk")

    # evaluate loaded model on test data
    loaded_modelTc1.tc1_net.eval()
    val_acc,val_loss = loaded_modelTc1.validation(0)

    print("Model State Dict Test loss: %5.2f"  % (val_loss))
    print("Model State Dict Test accuracy: %5.2f%%" %(val_acc))


def main():

    gParameters = initialize_parameters()
    run(gParameters)

if __name__ == '__main__':
    main()
    try:
        tmp = 1
    except AttributeError:      # theano does not have this function
        pass

