from __future__ import print_function

import numpy as np
import os
import sys

import torch

file_path = os.path.dirname(os.path.realpath(__file__))
lib_path2 = os.path.abspath(os.path.join(file_path, '..', '..', 'common'))
sys.path.append(lib_path2)
os.chdir(file_path)

import p1b2 as bmk
import candle
from torch_deps.p1b2_pytorch_model import P1B2Model
from torch_deps.random_seeding import seed_random_state

np.set_printoptions(precision=4)

def initialize_parameters(default_model = 'p1b2_default_model.txt'):

    # Build benchmark object
    p1b2Bmk = bmk.BenchmarkP1B2(bmk.file_path, default_model, 'pytorch',
    prog='p1b2_baseline', desc='Train Classifier - Pilot 1 Benchmark 2')

    print("Created P1B2 benchmark")
    
    # Initialize parameters
    gParameters = candle.finalize_parameters(p1b2Bmk)
    #benchmark.logger.info('Params: {}'.format(gParameters))
    print("Parameters initialized")

    return gParameters


def run(params):

    args = candle.ArgumentStruct(**params)
    args.no_cuda = args.no_cuda if hasattr(args,'no_cuda') else False
    args.multi_gpu = args.multi_gpu if hasattr(args,'multi_gpu') else True
    args.max_num_batches = args.max_num_batches if hasattr(args,'max_num_batches') else 1000
    args.dry_run = args.dry_run if hasattr(args,'dry_run') else False
    args.log_interval = args.log_interval if hasattr(args,'log_interval') else 10

    args.classes = args.classes if hasattr(args,'classes') else 10

    if args.loss=='categorical_crossentropy':
        args.out_activation='log_softmax'
        args.loss='nll'
        
    seed = args.rng_seed
    candle.set_seed(seed)
    # Setting up random seed for reproducible and deterministic results
    seed_random_state(args.rng_seed)

    args.keras_defaults = candle.keras_default_config()

    # Construct extension to save validation results
    ext = bmk.extension_from_parameters(params, '.pytorch')
    
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

    modelP1B2 = P1B2Model(args, use_cuda, device)
    
    #model.summary()
    #print(modelP1B2.p1b2_net)  # Model summary
    bmk.logger.info('Model summary: {}'.format(modelP1B2.p1b2_net))  # Model summary

    modelP1B2.train()
    modelP1B2.print_final_stats()

    #Save model 
    model_name = params['model_name']
    model_filename = "{}.model_state_dict.pth".format(params['save_path'])
    if hasattr(modelP1B2.p1b2_net,'module'):
        # Saving the DataParallel model
        torch.save(modelP1B2.p1b2_net.module.state_dict(), model_filename)
    else:
        torch.save(modelP1B2.p1b2_net.state_dict(), model_filename)
    
    #reload args from file
    args_file = open(args_filename, 'rb')
    loaded_args = pickle.load(args_file)
    args_file.close()

    # load weights into new model
    loaded_modelP1B2 = P1B2Model(loaded_args)
    loaded_modelP1B2.p1b2_net.load_state_dict(torch.load(model_filename, map_location=torch.device('cpu')))
    print("Loaded torch model from disk")

    # evaluate loaded model on test data
    loaded_modelP1B2.p1b2_net.eval()
    val_acc,val_loss = loaded_modelP1B2.validation(0)

    print("Model State Dict Validation loss: %5.2f"  % (val_loss))
    print("Model State Dict Validation accuracy: %5.2f%%" %(val_acc))

    print('Test data: ')
    test_acc,test_loss = loaded_modelP1B2.test()

    print("Model State Dict Test loss: %5.2f"  % (test_loss))
    print("Model State Dict Test accuracy: %5.2f%%" %(test_acc))
    
    
def main():

    gParameters = initialize_parameters()
    run(gParameters)

if __name__ == '__main__':
    main()
    try:
        tmp = 1
    except AttributeError:      # theano does not have this function
        pass


