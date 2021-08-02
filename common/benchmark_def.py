import argparse

try:
    import configparser
except ImportError:
    import ConfigParser as configparser

import os
import numpy as np
import inspect
import random

import parsing_utils
from helper_utils import eval_string_as_list_of_lists


def set_seed(seed):
    """Set the seed of the pseudo-random generator to the specified value.

        Parameters
        ----------
        seed : int
            Value to intialize or re-seed the generator.
    """
    os.environ['PYTHONHASHSEED'] = '0'
    np.random.seed(seed)

    random.seed(seed)


class Benchmark:
    """ Class that implements an interface to handle configuration options for the
        different CANDLE benchmarks.
        It provides access to all the common configuration
        options and configuration options particular to each individual benchmark.
        It describes what minimum requirements should be specified to instantiate
        the corresponding benchmark.
        It interacts with the argparser to extract command-line options and arguments
        from the benchmark's configuration files.
    """

    def __init__(self, filepath, defmodel, framework, prog=None, desc=None, parser=None):
        """ Initialize Benchmark object.

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

        self.registered_conf = []
        for lst in parsing_utils.registered_conf:
            self.registered_conf.extend(lst)

        self.required = set([])
        self.additional_definitions = []
        self.set_locals()

    def parse_parameters(self):
        """Functionality to parse options common
           for all benchmarks.
           This functionality is based on methods 'get_default_neon_parser' and
           'get_common_parser' which are defined previously(above). If the order changes
           or they are moved, the calling has to be updated.
        """
        # Parse has been split between arguments that are common with the default neon parser
        # and all the other options
        self.parser = parsing_utils.parse_common(self.parser)
        self.parser = parsing_utils.parse_from_dictlist(self.additional_definitions, self.parser)

        # Set default configuration file
        self.conffile = os.path.join(self.file_path, self.default_model)

    def format_benchmark_config_arguments(self, dictfileparam):
        """ Functionality to format the particular parameters of
            the benchmark.

            Parameters
            ----------
            dictfileparam : python dictionary
                parameters read from configuration file
            args : python dictionary
                parameters read from command-line
                Most of the time command-line overwrites configuration file
                except when the command-line is using default values and
                config file defines those values

        """

        configOut = dictfileparam.copy()
        kwall = self.additional_definitions + self.registered_conf

        for d in kwall:  # self.additional_definitions:
            if d['name'] in configOut.keys():
                if 'type' in d:
                    dtype = d['type']
                else:
                    dtype = None

                if 'action' in d:
                    if inspect.isclass(d['action']):
                        str_read = dictfileparam[d['name']]
                        configOut[d['name']] = eval_string_as_list_of_lists(str_read, ':', ',', dtype)
                elif d['default'] != argparse.SUPPRESS:
                    # default value on benchmark definition cannot overwrite config file
                    self.parser.add_argument('--' + d['name'],
                                             type=d['type'],
                                             default=configOut[d['name']],
                                             help=d['help'])

        return configOut

    def read_config_file(self, file):
        """Functionality to read the configue file
           specific for each benchmark.
        """

        config = configparser.ConfigParser()
        config.read(file)
        section = config.sections()
        fileParams = {}

        # parse specified arguments (minimal validation: if arguments
        # are written several times in the file, just the first time
        # will be used)
        for sec in section:
            for k, v in config.items(sec):
                # if not k in fileParams:
                if k not in fileParams:
                    fileParams[k] = eval(v)

        fileParams = self.format_benchmark_config_arguments(fileParams)

        print(fileParams)

        return fileParams

    def set_locals(self):
        """ Functionality to set variables specific for the benchmark
            - required: set of required parameters for the benchmark.
            - additional_definitions: list of dictionaries describing \
                the additional parameters for the benchmark.
        """

        pass

    def check_required_exists(self, gparam):
        """Functionality to verify that the required
           model parameters have been specified.
        """

        key_set = set(gparam.keys())
        intersect_set = key_set.intersection(self.required)
        diff_set = self.required.difference(intersect_set)

        if (len(diff_set) > 0):
            raise Exception(
                'ERROR ! Required parameters are not specified.  These required parameters have not been initialized: ' + str(sorted(diff_set)) + '... Exiting')
