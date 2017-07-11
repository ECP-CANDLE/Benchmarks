from __future__ import absolute_import

import numpy as np

import os
import sys
import gzip
import argparse
try:
    import configparser
except ImportError:
    import ConfigParser as configparser

file_path = os.path.dirname(os.path.realpath(__file__))
lib_path = os.path.abspath(os.path.join(file_path, '..', '..', 'common'))
sys.path.append(lib_path)

work_path = os.path.dirname(os.path.realpath(__file__))

from file_utils import get_file

# Seed for random generation -- default value
DEFAULT_SEED = 7102
DEFAULT_TIMEOUT = -1 # no timeout
DEFAULT_DATATYPE = np.float32


def fetch_file(link, subdir, untar=False, md5_hash=None):
    fname = os.path.basename(link)
    return get_file(fname, origin=link, untar=untar, md5_hash=md5_hash, cache_subdir=subdir)


def initialize_parameters(bmk):
    """Parse parameters in common and particular to each benchmark.
        
        Parameters
        ----------
        bmk : benchmark object
            Object that has benchmark filepaths and especifications
            
        Return
        ----------
        gParameters : dictionary with all the parameters necessary to run the benchmark.
            Command line overwrites config file especifications
    """

    # Parse common parameters
    bmk.parse_from_common()
    # Parse parameters that are applicable just to benchmark
    bmk.parse_from_benchmark()

    # Get command-line parameters
    args = bmk.parser.parse_args()
    #print('Args:', args)
    # Get parameters from configuration file
    fileParameters = bmk.read_config_file(args.config_file)
    #print ('Params:', fileParameters)
    # Consolidate parameter set. Command-line parameters overwrite file configuration
    gParameters = args_overwrite_config(args, fileParameters)
    print ('Params:', gParameters)
    
    return gParameters


def get_default_neon_parser(parser):
    """Parse command-line arguments that are default in neon parser (and are common to all frameworks). 
        Ignore if not present.
        
        Parameters
        ----------
        parser : python argparse
            parser for neon default command-line options
    """
    # Logging Level
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="increase output verbosity")
    parser.add_argument("-l", "--log", dest='logfile',
                        default=None,
                        help="log file")
                        
    # Logging utilities
    parser.add_argument("-s", "--save_path", dest='save_path',
                        default=argparse.SUPPRESS, type=str,
                        help="file path to save model snapshots")

    # General behavior
    parser.add_argument("--model_file", dest='weight_path', type=str,
                        default=argparse.SUPPRESS,
                        help="specify trained model Pickle file")
    parser.add_argument("-d", "--data_type", dest='datatype',
                        default=argparse.SUPPRESS,
                        choices=['f16', 'f32', 'f64'],
                        help="default floating point")

    # Model definition
    # Model Architecture
    parser.add_argument("--dense", nargs='+', type=int,
                        default=argparse.SUPPRESS,
                        help="number of units in fully connected layers in an integer array")

    # Data preprocessing
    #parser.add_argument("--shuffle", action="store_true",
    #                    default=True,
    #                    help="randomly shuffle data set (produces different training and testing partitions each run depending on the seed)")

    # Training configuration
    parser.add_argument("-r", "--rng_seed", dest='rng_seed', type=int,
                        default=argparse.SUPPRESS,
                        help="random number generator seed")
    parser.add_argument("-e", "--epochs", type=int,
                        default=argparse.SUPPRESS,
                        help="number of training epochs")
    parser.add_argument("-z", "--batch_size", type=int,
                        default=argparse.SUPPRESS,
                        help="batch size")

    return parser


def get_common_parser(parser):
    """Parse command-line arguments. Ignore if not present.
        
        Parameters
        ----------
        parser : python argparse
            parser for command-line options
    """
    
    # General behavior
    parser.add_argument("--train", dest='train_bool', action="store_true",
                        default=True, #type=bool,
                        help="train model")
    parser.add_argument("--evaluate", dest='eval_bool', action="store_true",
                        default=argparse.SUPPRESS, #type=bool,
                        help="evaluate model (use it for inference)")

    parser.add_argument("--timeout", dest='timeout', action="store",
                    default=argparse.SUPPRESS,
                    help="seconds allowed to train model (default: no timeout)")


    # Logging utilities
    parser.add_argument("--home_dir", dest='home_dir',
                        default=argparse.SUPPRESS, type=str,
                        help="set home directory")
                        
    parser.add_argument("--train_data", action="store",
                        default=argparse.SUPPRESS,
                        help="training data filename")

    parser.add_argument("--test_data", action="store",
                        default=argparse.SUPPRESS,
                        help="testing data filename")

    parser.add_argument("--output_dir", dest='output_dir',
                        default=argparse.SUPPRESS, type=str,
                        help="output directory")
                        
    parser.add_argument("--data_url", dest='data_url',
                        default=argparse.SUPPRESS, type=str,
                        help="set data source url")

    parser.add_argument("--experiment_id", default="EXP000", type=str, help="set the experiment unique identifier")

    parser.add_argument("--run_id", default="RUN000", type=str, help="set the run unique identifier")
    


    # Model definition
    # Model Architecture
    parser.add_argument("--conv", nargs='+', type=int,
                        default=argparse.SUPPRESS,
                        help="integer array describing convolution layers: conv1_filters, conv1_filter_len, conv1_stride, conv2_filters, conv2_filter_len, conv2_stride ...")
    parser.add_argument("--locally_connected", action="store_true",
                        default=argparse.SUPPRESS,
                        help="use locally connected layers instead of convolution layers")
    parser.add_argument("-a", "--activation",
                        default=argparse.SUPPRESS,
                        help="keras activation function to use in inner layers: relu, tanh, sigmoid...")
    parser.add_argument("--out_activation",
                        default=argparse.SUPPRESS,
                        help="keras activation function to use in out layer: softmax, linear, ...")
                        
                        
    parser.add_argument("--lstm_size", nargs='+', type=int,
                        default= argparse.SUPPRESS,
                        help="integer array describing size of LSTM internal state per layer")
    parser.add_argument("--recurrent_dropout", action="store",
                        default=argparse.SUPPRESS, type=float,
                        help="ratio of recurrent dropout")
                        
                        
    # Processing between layers
    parser.add_argument("--drop", type=float,
                        default=argparse.SUPPRESS,
                        help="ratio of dropout used in fully connected layers")
    parser.add_argument("--pool", type=int,
                        default=argparse.SUPPRESS,
                        help="pooling layer length")
    parser.add_argument("--batch_normalization", action="store_true",
                        default=argparse.SUPPRESS,
                        help="use batch normalization")
                        
    # Model Evaluation
    parser.add_argument("--loss",
                        default=argparse.SUPPRESS,
                        help="keras loss function to use: mse, ...")
    parser.add_argument("--optimizer",
                        default=argparse.SUPPRESS,
                        help="keras optimizer to use: sgd, rmsprop, ...")

    parser.add_argument("--metrics",
                        default=argparse.SUPPRESS,
                        help="metrics to evaluate performance: accuracy, ...")
    
    # Data preprocessing
    parser.add_argument("--scaling",
                        default=argparse.SUPPRESS,
                        choices=['minabs', 'minmax', 'std', 'none'],
                        help="type of feature scaling; 'minabs': to [-1,1]; 'minmax': to [0,1], 'std': standard unit normalization; 'none': no normalization")
    parser.add_argument("--shuffle", action="store_true",
                        default=True,
                        help="randomly shuffle data set (produces different training and testing partitions each run depending on the seed)")

    # Feature selection
    parser.add_argument("--feature_subsample", type=int,
                        default=argparse.SUPPRESS,
                        help="number of features to randomly sample from each category (cellline expression, drug descriptors, etc), 0 means using all features")

    # Training configuration
    parser.add_argument("--learning_rate",
                        default= argparse.SUPPRESS, type=float,
                        help="overrides the learning rate for training")
    
    parser.add_argument("--initialization",
                        default=argparse.SUPPRESS,
                        choices=['constant', 'uniform', 'normal', 'glorot_uniform', 'lecun_uniform', 'he_normal'],
                        help="type of weight initialization; 'constant': to 0; 'uniform': to [-0.05,0.05], 'normal': mean 0, stddev 0.05; 'glorot_uniform': [-lim,lim] with lim = sqrt(6/(fan_in+fan_out)); 'lecun_uniform' : [-lim,lim] with lim = sqrt(3/fan_in); 'he_normal' : mean 0, stddev sqrt(2/fan_in)")
    parser.add_argument("--val_split", type=float,
                        default=argparse.SUPPRESS,
                        help="fraction of data to use in validation")
    parser.add_argument("--train_steps", type=int,
                        default=argparse.SUPPRESS,
                        help="overrides the number of training batches per epoch if set to nonzero")
    parser.add_argument("--val_steps", type=int,
                        default=argparse.SUPPRESS,
                        help="overrides the number of validation batches per epoch if set to nonzero")
    parser.add_argument("--test_steps", type=int,
                        default=argparse.SUPPRESS,
                        help="overrides the number of test batches per epoch if set to nonzero")
    parser.add_argument("--train_samples", action="store",
                        default=argparse.SUPPRESS, type=int,
                        help="overrides the number of training samples if set to nonzero")
    parser.add_argument("--val_samples", action="store",
                        default=argparse.SUPPRESS, type=int,
                        help="overrides the number of validation samples if set to nonzero")
    
    
    # Backend configuration
    parser.add_argument("--gpus", action="store", nargs='*',
                        default=[], type=int,
                        help="set IDs of GPUs to use")


    return parser



def args_overwrite_config(args, config):
    """Overwrite configuration parameters with 
        parameters specified via command-line.
        
        Parameters
        ----------
        args : python argparse
            parameters specified via command-line
        config : python dictionary
            parameters read from configuration file
    """
    
    params = config
    
    args_dict = vars(args)
    
    for key in args_dict.keys():
        params[key] = args_dict[key]
    
    
    if 'datatype' not in params:
        params['datatype'] = DEFAULT_DATATYPE
    else:
        if params['datatype'] in set(['f16', 'f32', 'f64']):
            params['datatype'] = get_choice(params['datatype'])

    if 'output_dir' not in params:
        params['output_dir'] = directory_from_parameters(params)
    else:
        params['output_dir'] = directory_from_parameters(params, params['output_dir'])

    if 'rng_seed' not in params:
        params['rng_seed'] = DEFAULT_SEED

    if 'timeout' not in params:
        params['timeout'] = DEFAULT_TIMEOUT


    return params



def get_choice(name):
    """ Maps name string to the right type of argument
    """
    mapping = {}
    
    # dtype
    mapping['f16'] = np.float16
    mapping['f32'] = np.float32
    mapping['f64'] = np.float64
    
    mapped = mapping.get(name)
    if not mapped:
        raise Exception('No mapping found for "{}"'.format(name))
    
    return mapped


def directory_from_parameters(params, commonroot='Output'):
    """Construct output directory path with unique IDs from parameters"""
    
    if commonroot in set(['.', './']): # Same directory --> convert to absolute path
        outdir = os.path.abspath('.')
    else: # Create path specified
        outdir = os.path.abspath(os.path.join('.', commonroot))
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        outdir = os.path.abspath(os.path.join(outdir, params['experiment_id']))
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        outdir = os.path.abspath(os.path.join(outdir, params['run_id']))
        if not os.path.exists(outdir):
            os.makedirs(outdir)


    return outdir



class Benchmark:

    def __init__(self, filepath, defmodel, framework, prog=None, desc=None, parser=None):
        """Initialize benchmark object. Object to group common and 
        specific (to benchmark) configuration options.
        
            Parameters
            ----------
            filepath : ./
                os.path.dirname where the benchmark is located. Necessary to locate utils and
                establish input/ouput paths
            defmodel : 'p*b*_default_model.txt'
                string corresponding to the default model of the benchmark
            framework : 'keras', 'neon', 'mxnet', 'pytorch'
                framework used to run the benchmark
            prog : 'p*b*_baseline_*'
                string for program name (usually associated to benchmark and framework)
            desc : ' '
                string describing benchmark (usually a description of the neural network model built)
            parser : argparser (default None)
                if 'neon' framework a NeonArgparser is passed. Otherwise an argparser is constructed.
        """
        
        if parser is None:
            parser = argparse.ArgumentParser(prog=prog, formatter_class=argparse.ArgumentDefaultsHelpFormatter, description=desc, conflict_handler='resolve')

        self.parser = parser
        self.file_path = filepath
        self.default_model = defmodel
        self.framework = framework


    def parse_from_common(self):
        """Functionality to parse options common
           for all benchmarks.
           This functionality is based on methods 'get_default_neon_parser' and
           'get_common_parser' which are defined previously(above). If the order changes
           or they are moved, the calling has to be updated.
        """
        
        # Set default configuration file
        self.parser.add_argument("--config_file", dest='config_file', type=str,
                        default=os.path.join(self.file_path, self.default_model),
                        help="specify model configuration file")
                        
        # Parse has been split between arguments that are common with the default neon parser
        # and all the other options
        parser = self.parser
        if self.framework is not 'neon':
            parser = get_default_neon_parser(parser)
        parser = get_common_parser(parser)
    
        self.parser = parser
    
    
    def parse_from_benchmark(self):
        """Functionality to parse options specific
           specific for each benchmark.
        """
        
        raise NotImplementedError()
    

    def read_config_file(self, file):
        """Functionality to read the configue file
           specific for each benchmark.
        """

        raise NotImplementedError()



def keras_default_config():
    """Defines parameters that intervine in different functions using the keras defaults.
        This helps to keep consistency in parameters between frameworks.
    """
    
    kerasDefaults = {}
    
    # Optimizers
    #kerasDefaults['clipnorm']=?            # Maximum norm to clip all parameter gradients
    #kerasDefaults['clipvalue']=?          # Maximum (minimum=-max) value to clip all parameter gradients
    kerasDefaults['decay_lr']=0.            # Learning rate decay over each update
    kerasDefaults['epsilon']=1e-8           # Factor to avoid divide by zero (fuzz factor)
    kerasDefaults['rho']=0.9                # Decay parameter in some optmizer updates (rmsprop, adadelta)
    kerasDefaults['momentum_sgd']=0.        # Momentum for parameter update in sgd optimizer
    kerasDefaults['nesterov_sgd']=False     # Whether to apply Nesterov momentum in sgd optimizer
    kerasDefaults['beta_1']=0.9             # Parameter in some optmizer updates (adam, adamax, nadam)
    kerasDefaults['beta_2']=0.999           # Parameter in some optmizer updates (adam, adamax, nadam)
    kerasDefaults['decay_schedule_lr']=0.004# Parameter for nadam optmizer

    # Initializers
    kerasDefaults['minval_uniform']=-0.05   #  Lower bound of the range of random values to generate
    kerasDefaults['maxval_uniform']=0.05    #  Upper bound of the range of random values to generate
    kerasDefaults['mean_normal']=0.         #  Mean of the random values to generate
    kerasDefaults['stddev_normal']=0.05     #  Standard deviation of the random values to generate


    return kerasDefaults

