import os
import sys


file_path = os.path.dirname(os.path.realpath(__file__))
lib_path = os.path.abspath(os.path.join(file_path, '..'))
sys.path.append(lib_path)
lib_path2 = os.path.abspath(os.path.join(file_path, '..', '..', 'common', 'darts'))
sys.path.append(lib_path2)


import darts
#from darts.architecture import Architecture

print(Architecture)
print(darts.architecture.Architecture)
print(darts.ConvNetwork)
print(darts.Genotype)
