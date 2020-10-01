"""
    File Name:          UnoPytorch/tee.py
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               8/17/18
    Python Version:     3.6.6
    File Description:
        This file implements a helper class Tee, which redirects the stdout
        to a file while keeping things printed in console.
"""
import os
import sys


class Tee(object):
    """Tee class for storing terminal output to files.

    This class implements a tee class that flush std terminal output to a
    file for logging purpose.
    """

    def __init__(self, log_name, mode='a'):

        self.__stdout = sys.stdout

        self.__log_name = log_name
        self.__mode = mode

        try:
            os.makedirs(os.path.dirname(log_name))
        except FileExistsError:
            pass

    def __del__(self):
        sys.stdout = self.__stdout

    def write(self, data):

        with open(self.__log_name, self.__mode) as file:
            file.write(data)

        self.__stdout.write(data)

    def flush(self):
        self.__stdout.flush()

    def default_stdout(self):
        return self.__stdout
